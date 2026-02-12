import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

def _as_layer_tuple(value):
    if value is None:
        return tuple()
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    return tuple()


def _sanitize_intervals(intervals):
    result = {}
    if not isinstance(intervals, dict):
        return result

    for key, val in intervals.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            start, end = int(val[0]), int(val[1])
            if end > start:
                result[str(key)] = (start, end)

    return result


def _normalize_attention_payload(attentions):
    payload = {
        "encoder_fusion": tuple(),
        "decoder": tuple(),
        "token_intervals": {
            "encoder_fusion": {},
            "decoder": {},
        },
    }

    if attentions is None:
        return payload

    if isinstance(attentions, dict):
        payload["encoder_fusion"] = _as_layer_tuple(attentions.get("encoder_fusion"))
        payload["decoder"] = _as_layer_tuple(attentions.get("decoder"))

        token_intervals = attentions.get("token_intervals", {})
        if isinstance(token_intervals, dict):
            payload["token_intervals"]["encoder_fusion"] = _sanitize_intervals(token_intervals.get("encoder_fusion", {}))
            payload["token_intervals"]["decoder"] = _sanitize_intervals(token_intervals.get("decoder", {}))

        return payload

    payload["decoder"] = _as_layer_tuple(attentions)
    return payload


def _to_sample_attention_matrices(layer_attention: torch.Tensor) -> torch.Tensor:
    att = layer_attention.detach().float().cpu()

    if att.dim() == 4:
        # [B, heads, q_tokens, k_tokens] -> [B, q_tokens, k_tokens]
        att = att.mean(dim=1)
    elif att.dim() == 3:
        # [B, q_tokens, k_tokens]
        pass
    elif att.dim() == 2:
        # [q_tokens, k_tokens] -> [1, q_tokens, k_tokens]
        att = att.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected attention tensor shape: {tuple(att.shape)}")

    return att


def _default_decoder_token_intervals(model):
    intervals = {}
    start = 1  # decoder CLS is index 0
    for modality in model.modalities:
        end = start + model.get_patches_number(modality)
        intervals[modality] = (start, end)
        start = end
    return intervals


def _collect_attention_heatmaps(
    model,
    dataloader,
    device: str,
    split_name: str = "test",
    zero_rna: bool = False,
    permute_rna: bool = False,
    ATTN_SOURCES : tuple[str, ...] = ("encoder_fusion", "decoder"),
):
    if zero_rna and permute_rna:
        raise ValueError("Choose only one RNA perturbation mode: zero_rna or permute_rna.")

    raw_attention_payloads = []
    layer_sums = {source: [] for source in ATTN_SOURCES}
    layer_counts = {source: [] for source in ATTN_SOURCES}
    token_intervals = {source: {} for source in ATTN_SOURCES}

    if zero_rna:
        desc = f"{split_name} inference | RNA=0"
    elif permute_rna:
        desc = f"{split_name} inference | RNA=perm"
    else:
        desc = f"{split_name} inference | RNA=orig"
    model.eval()

    with torch.no_grad():
        for values_batch, masks_batch in tqdm.tqdm(dataloader, desc=desc):
            values_batch = {modality: value.to(device) for modality, value in values_batch.items()}
            masks_batch = {modality: value.to(device) for modality, value in masks_batch.items()}

            if zero_rna and "rna" in values_batch:
                values_batch = dict(values_batch)
                values_batch["rna"] = torch.zeros_like(values_batch["rna"])
            elif permute_rna and "rna" in values_batch:
                values_batch = dict(values_batch)
                batch_size = values_batch["rna"].shape[0]
                permutation = torch.randperm(batch_size, device=values_batch["rna"].device)
                values_batch["rna"] = values_batch["rna"][permutation]

            outputs = model(values_batch, masks_batch)
            payload = _normalize_attention_payload(outputs.attentions)

            payload_cpu = {
                "encoder_fusion": tuple(att.detach().cpu() for att in payload["encoder_fusion"]),
                "decoder": tuple(att.detach().cpu() for att in payload["decoder"]),
                "token_intervals": {
                    "encoder_fusion": dict(payload["token_intervals"]["encoder_fusion"]),
                    "decoder": dict(payload["token_intervals"]["decoder"]),
                },
            }
            raw_attention_payloads.append(payload_cpu)

            for source in ATTN_SOURCES:
                layers = payload_cpu[source]
                if len(layers) == 0:
                    continue

                if not token_intervals[source]:
                    token_intervals[source] = dict(payload_cpu["token_intervals"].get(source, {}))

                if len(layer_sums[source]) == 0:
                    layer_sums[source] = [None] * len(layers)
                    layer_counts[source] = [0] * len(layers)
                elif len(layers) != len(layer_sums[source]):
                    raise RuntimeError(
                        f"Number of {source} attention layers changed across batches: "
                        f"{len(layer_sums[source])} -> {len(layers)}"
                    )

                for layer_idx, layer_att in enumerate(layers):
                    sample_mats = _to_sample_attention_matrices(layer_att)
                    batch_sum = sample_mats.sum(dim=0).to(torch.float64)

                    if layer_sums[source][layer_idx] is None:
                        layer_sums[source][layer_idx] = batch_sum
                    else:
                        if layer_sums[source][layer_idx].shape != batch_sum.shape:
                            raise RuntimeError(
                                f"Attention shape mismatch in {source}, layer {layer_idx + 1}: "
                                f"{tuple(layer_sums[source][layer_idx].shape)} vs {tuple(batch_sum.shape)}"
                            )
                        layer_sums[source][layer_idx] += batch_sum

                    layer_counts[source][layer_idx] += sample_mats.shape[0]

    if len(raw_attention_payloads) == 0:
        if zero_rna:
            mode = "RNA=0"
        elif permute_rna:
            mode = "RNA=perm"
        else:
            mode = "RNA=orig"
        raise RuntimeError(
            f"No attention payloads were found in outputs.attentions on the '{split_name}' dataset ({mode})."
        )

    mean_heatmaps = {source: [] for source in ATTN_SOURCES}
    for source in ATTN_SOURCES:
        if len(layer_sums[source]) == 0:
            continue

        mean_heatmaps[source] = [
            (layer_sums[source][i] / layer_counts[source][i]).to(torch.float32)
            for i in range(len(layer_sums[source]))
        ]

    if not token_intervals["decoder"]:
        token_intervals["decoder"] = _default_decoder_token_intervals(model)
    if not token_intervals["encoder_fusion"]:
        token_intervals["encoder_fusion"] = dict(token_intervals["decoder"])

    return raw_attention_payloads, mean_heatmaps, token_intervals

def _format_modality_ranges(
    intervals, 
    MODALITY_ORDER,
    MODALITY_NAMES,
):
    parts = ["CLS[0]"]
    for modality in MODALITY_ORDER:
        if modality not in intervals:
            continue
        start, end = intervals[modality]
        parts.append(f"{MODALITY_NAMES.get(modality, modality.upper())}[{start}:{end - 1}]")
    return " | ".join(parts)


def _draw_modality_boundaries(
    ax, 
    intervals, 
    matrix_size,
    MODALITY_ORDER,
    MODALITY_COLORS,
):
    for modality in MODALITY_ORDER:
        if modality not in intervals:
            continue

        color = MODALITY_COLORS.get(modality, "white")
        start, end = intervals[modality]

        start = max(0, min(matrix_size, start))
        end = max(0, min(matrix_size, end))

        for pos in (start, end):
            line_pos = pos - 0.5
            if -0.5 <= line_pos <= matrix_size - 0.5:
                ax.axvline(line_pos, color=color, linestyle="--", linewidth=0.8, alpha=0.95)
                ax.axhline(line_pos, color=color, linestyle="--", linewidth=0.8, alpha=0.95)


def _percentile_bounds(values_1d: np.ndarray, low_q: float = 1.0, high_q: float = 99.0):
    if values_1d.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(values_1d, low_q))
    vmax = float(np.percentile(values_1d, high_q))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0

    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    return vmin, vmax


def _format_token_description(
    modality: str,
    local_token_index: int,
    token_description: str | None,
    modality_names: dict[str, str],
    max_chars: int = 56,
) -> str:
    modality_name = modality_names.get(modality, modality.upper())
    if token_description is None:
        return f"{modality_name} token {local_token_index}"

    clean_description = " ".join(str(token_description).split())
    if len(clean_description) > max_chars:
        clean_description = clean_description[: max_chars - 3] + "..."
    return f"{modality_name}: {clean_description}"


def _get_top_key_token_ticks(
    key_scores: np.ndarray,
    intervals: dict[str, tuple[int, int]],
    token_index_to_description: dict[int, str] | None,
    modality_order: tuple[str, ...],
    modality_names: dict[str, str],
    top_k_per_modality: int = 5,
) -> tuple[list[int], list[str]]:
    if key_scores.size == 0:
        return [], []

    token_index_to_description = token_index_to_description or {}
    focused_modalities = [m for m in modality_order if m in ("rna", "dnam")]
    tick_pairs: list[tuple[int, str]] = []

    for modality in focused_modalities:
        if modality not in intervals:
            continue

        start, end = intervals[modality]
        start = int(max(0, min(start, key_scores.shape[0])))
        end = int(max(0, min(end, key_scores.shape[0])))
        if end <= start:
            continue

        modality_scores = key_scores[start:end]
        k = min(top_k_per_modality, modality_scores.shape[0])
        if k <= 0:
            continue

        top_local_indices = np.argsort(modality_scores)[-k:][::-1]
        for local_idx in top_local_indices:
            token_idx = start + int(local_idx)
            description = token_index_to_description.get(token_idx)
            label = _format_token_description(
                modality=modality,
                local_token_index=int(local_idx),
                token_description=description,
                modality_names=modality_names,
            )
            tick_pairs.append((token_idx, label))

    if len(tick_pairs) == 0:
        return [], []

    # Sort by token position for easier visual matching on the x axis.
    tick_pairs = sorted(tick_pairs, key=lambda pair: pair[0])
    tick_positions = [pair[0] for pair in tick_pairs]
    tick_labels = [pair[1] for pair in tick_pairs]
    return tick_positions, tick_labels


def _draw_diagonal_token_callouts(
    ax,
    tick_positions: list[int],
    tick_labels: list[str],
    matrix_size: int,
):
    if len(tick_positions) == 0:
        return

    y_offset = -0.8
    x0, x1 = ax.get_xlim()  # heatmap span in data units, usually -0.5 .. matrix_size-0.5
    label_xs = np.linspace(x0 + 0.5, x1 - 1, len(tick_positions))  # evenly spaced

    # position labels
    for token_idx, label, label_x in zip(tick_positions, tick_labels, label_xs):
        ax.annotate(
            label,
            xy=(token_idx, matrix_size - 0.5),
            xycoords="data",
            xytext=(label_x, y_offset),
            textcoords=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
            rotation=90,
            rotation_mode="anchor",
            color="black",
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.5,
                color="black",
                # Anchor arrow from the top side of the rotated label box, not its center.
                relpos=(0.5, 1.0),
                shrinkA=0,
                shrinkB=0,
            ),
            annotation_clip=False,
            clip_on=False,
        )


def _plot_attention_source_comparison(
    source_name: str,
    mean_heatmaps_present,
    mean_heatmaps_zeroed,
    mean_heatmaps_permuted,
    intervals,
    token_index_to_description: dict[int, str] | None = None,
    top_k_per_modality: int = 5,
    MODALITY_ORDER : tuple[str, ...] = ("rna", "dnam", "wsi"),
    MODALITY_NAMES : dict[str, str] = {
        "rna": "RNA",
        "dnam": "DNA",
        "wsi": "WSI",
    },
    MODALITY_COLORS : dict[str, str] = {
        "rna": "deepskyblue",
        "dnam": "limegreen",
        "wsi": "orange",
    },
):
    if len(mean_heatmaps_present) == 0 and len(mean_heatmaps_zeroed) == 0 and len(mean_heatmaps_permuted) == 0:
        print(f"No attentions found for source='{source_name}'.")
        return

    if len(mean_heatmaps_present) != len(mean_heatmaps_zeroed):
        raise RuntimeError(
            f"Different number of {source_name} layers between RNA-present and RNA-zeroed runs: "
            f"{len(mean_heatmaps_present)} vs {len(mean_heatmaps_zeroed)}"
        )
    if len(mean_heatmaps_present) != len(mean_heatmaps_permuted):
        raise RuntimeError(
            f"Different number of {source_name} layers between RNA-present and RNA-permuted runs: "
            f"{len(mean_heatmaps_present)} vs {len(mean_heatmaps_permuted)}"
        )

    num_layers = len(mean_heatmaps_present)
    deltas_zeroed = [mean_heatmaps_zeroed[i] - mean_heatmaps_present[i] for i in range(num_layers)]
    deltas_permuted = [mean_heatmaps_permuted[i] - mean_heatmaps_present[i] for i in range(num_layers)]

    common_values = torch.cat(
        [x.flatten() for x in (mean_heatmaps_present + mean_heatmaps_zeroed + mean_heatmaps_permuted)]
    ).detach().cpu().numpy()
    common_vmin, common_vmax = _percentile_bounds(common_values, low_q=1.0, high_q=99.0)

    delta_values = torch.cat([d.flatten() for d in (deltas_zeroed + deltas_permuted)]).detach().cpu().numpy()
    delta_vmin, delta_vmax = _percentile_bounds(delta_values, low_q=1.0, high_q=99.0)
    if delta_vmax <= 0:
        delta_vmax = 1e-12
    if delta_vmin >= 0:
        delta_vmin = -1e-12

    fig, axes = plt.subplots(num_layers, 5, figsize=(20, 7 * num_layers))
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    # Reserve larger vertical gaps between rows for diagonal token callouts.
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.90, wspace=0.05, hspace=1.75)

    source_title = "Encoder Fusion" if source_name == "encoder_fusion" else "Decoder"
    interval_text = _format_modality_ranges(intervals, MODALITY_ORDER=MODALITY_ORDER, MODALITY_NAMES=MODALITY_NAMES)
    fig.suptitle(f"{source_title} attentions\nToken ranges: {interval_text}", fontsize=13, y=0.97)

    im0 = None
    im2 = None

    for layer_idx in range(num_layers):
        heat_present = mean_heatmaps_present[layer_idx].numpy()
        heat_zeroed = mean_heatmaps_zeroed[layer_idx].numpy()
        heat_permuted = mean_heatmaps_permuted[layer_idx].numpy()
        heat_delta_zeroed = deltas_zeroed[layer_idx].numpy()
        heat_delta_permuted = deltas_permuted[layer_idx].numpy()

        matrix_size = heat_present.shape[0]
        top_ticks_by_name: dict[str, tuple[list[int], list[str]]] = {}
        key_scores_by_name = {
            "present": heat_present.mean(axis=0),
            "zeroed": heat_zeroed.mean(axis=0),
            "permuted": heat_permuted.mean(axis=0),
        }
        for heatmap_name, key_scores in key_scores_by_name.items():
            top_ticks_by_name[heatmap_name] = _get_top_key_token_ticks(
                key_scores=key_scores,
                intervals=intervals,
                token_index_to_description=token_index_to_description,
                modality_order=MODALITY_ORDER,
                modality_names=MODALITY_NAMES,
                top_k_per_modality=top_k_per_modality,
            )
        top_ticks_by_col = {
            0: top_ticks_by_name["present"],
            1: top_ticks_by_name["zeroed"],
            2: top_ticks_by_name["permuted"],
            3: top_ticks_by_name["zeroed"],
            4: top_ticks_by_name["permuted"],
        }

        im0 = axes[layer_idx, 0].imshow(
            heat_present,
            cmap="viridis",
            vmin=common_vmin,
            vmax=common_vmax,
            aspect="equal",
        )
        im1 = axes[layer_idx, 1].imshow(
            heat_zeroed,
            cmap="viridis",
            vmin=common_vmin,
            vmax=common_vmax,
            aspect="equal",
        )
        im2 = axes[layer_idx, 2].imshow(
            heat_permuted,
            cmap="viridis",
            vmin=common_vmin,
            vmax=common_vmax,
            aspect="equal",
        )
        im3 = axes[layer_idx, 3].imshow(
            heat_delta_zeroed,
            cmap="coolwarm",
            vmin=delta_vmin,
            vmax=delta_vmax,
            aspect="equal",
        )
        im4 = axes[layer_idx, 4].imshow(
            heat_delta_permuted,
            cmap="coolwarm",
            vmin=delta_vmin,
            vmax=delta_vmax,
            aspect="equal",
        )

        axes[layer_idx, 0].set_title(f"Layer {layer_idx + 1}: RNA present", pad=8)
        axes[layer_idx, 1].set_title(f"Layer {layer_idx + 1}: RNA zeroed", pad=8)
        axes[layer_idx, 2].set_title(f"Layer {layer_idx + 1}: RNA permuted", pad=8)
        axes[layer_idx, 3].set_title(f"Layer {layer_idx + 1}: Zeroed - Present", pad=8)
        axes[layer_idx, 4].set_title(f"Layer {layer_idx + 1}: Permuted - Present", pad=8)

        for col_idx in range(5):
            ax = axes[layer_idx, col_idx]
            ax.set_ylabel("Query tokens")
            ax.set_xlabel("Key tokens")
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            top_tick_positions, top_tick_labels = top_ticks_by_col.get(col_idx, ([], []))
            if len(top_tick_positions) > 0:
                # ax.set_xticks(top_tick_positions)
                # ax.set_xticklabels([str(idx) for idx in top_tick_positions], fontsize=6)
                _draw_diagonal_token_callouts(
                    ax=ax,
                    tick_positions=top_tick_positions,
                    tick_labels=top_tick_labels,
                    matrix_size=matrix_size,
                )
            _draw_modality_boundaries(
                ax, 
                intervals, 
                matrix_size,
                MODALITY_ORDER = MODALITY_ORDER,
                MODALITY_COLORS = MODALITY_COLORS,
            )

    # Dedicated colorbar axes placed on opposite sides of the heatmap grid.
    cax_main = fig.add_axes([0.00, 0.22, 0.014, 0.58])
    cax_delta = fig.add_axes([0.94, 0.22, 0.014, 0.58])

    if im0 is not None:
        cb_main = fig.colorbar(im0, cax=cax_main)
        cb_main.set_label("Attention score")
    if im4 is not None:
        cb_delta = fig.colorbar(im4, cax=cax_delta)
        cb_delta.set_label("Delta")

    # plt.show()
