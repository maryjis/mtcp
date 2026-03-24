import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm

##################################################################
# attention maps collection
##################################################################

_RNA_ATTENTION_MODES = ("rna_present", "rna_zeroed", "rna_permuted")

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


def _compute_batch_attention_sum(layer_attention: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Reduce one attention layer to a CPU batch sum and sample count.

    The reduction is performed on the source device first (including head
    averaging for 4D tensors), then moved to CPU as float64 for stable
    cross-batch accumulation.
    """
    att = layer_attention.detach().float()

    if att.dim() == 4:
        # [B, heads, q_tokens, k_tokens] -> sum over batch of mean-over-heads maps.
        sample_count = int(att.shape[0])
        batch_sum = att.mean(dim=1).sum(dim=0)
    elif att.dim() == 3:
        # [B, q_tokens, k_tokens] -> sum over batch.
        sample_count = int(att.shape[0])
        batch_sum = att.sum(dim=0)
    elif att.dim() == 2:
        # [q_tokens, k_tokens] -> one-sample batch sum.
        sample_count = 1
        batch_sum = att
    else:
        raise ValueError(f"Unexpected attention tensor shape: {tuple(att.shape)}")

    return batch_sum.to(device="cpu", dtype=torch.float64), sample_count


def _init_attention_state(
    ATTN_SOURCES: tuple[str, ...],
    collect_raw_attention_payloads: bool,
):
    """Initialize mutable accumulators used across batches for each source."""
    return {
        "raw_attention_payloads": [] if collect_raw_attention_payloads else None,
        "layer_sums": {source: [] for source in ATTN_SOURCES},
        "layer_counts": {source: [] for source in ATTN_SOURCES},
        "token_intervals": {source: {} for source in ATTN_SOURCES},
        "found_attention_payload": False,
    }


def _accumulate_attention_payload(
    payload,
    state,
    ATTN_SOURCES: tuple[str, ...],
):
    """Accumulate one normalized attention payload into the running state."""
    if state["raw_attention_payloads"] is not None:
        payload_cpu = {
            "encoder_fusion": tuple(att.detach().cpu() for att in payload["encoder_fusion"]),
            "decoder": tuple(att.detach().cpu() for att in payload["decoder"]),
            "token_intervals": {
                "encoder_fusion": dict(payload["token_intervals"]["encoder_fusion"]),
                "decoder": dict(payload["token_intervals"]["decoder"]),
            },
        }
        state["raw_attention_payloads"].append(payload_cpu)

    layer_sums = state["layer_sums"]
    layer_counts = state["layer_counts"]
    token_intervals = state["token_intervals"]

    for source in ATTN_SOURCES:
        layers = payload[source]
        if len(layers) == 0:
            continue

        state["found_attention_payload"] = True

        if not token_intervals[source]:
            token_intervals[source] = dict(payload["token_intervals"].get(source, {}))

        if len(layer_sums[source]) == 0:
            layer_sums[source] = [None] * len(layers)
            layer_counts[source] = [0] * len(layers)
        elif len(layers) != len(layer_sums[source]):
            raise RuntimeError(
                f"Number of {source} attention layers changed across batches: "
                f"{len(layer_sums[source])} -> {len(layers)}"
            )

        for layer_idx, layer_att in enumerate(layers):
            batch_sum, sample_count = _compute_batch_attention_sum(layer_att)

            if layer_sums[source][layer_idx] is None:
                layer_sums[source][layer_idx] = batch_sum
            else:
                if layer_sums[source][layer_idx].shape != batch_sum.shape:
                    raise RuntimeError(
                        f"Attention shape mismatch in {source}, layer {layer_idx + 1}: "
                        f"{tuple(layer_sums[source][layer_idx].shape)} vs {tuple(batch_sum.shape)}"
                    )
                layer_sums[source][layer_idx] += batch_sum

            layer_counts[source][layer_idx] += sample_count


def _finalize_attention_state(
    state,
    model,
    split_name: str,
    mode_label: str,
    ATTN_SOURCES: tuple[str, ...],
):
    """Finalize accumulated state into mean heatmaps and token intervals."""
    if not state["found_attention_payload"]:
        raise RuntimeError(
            f"No attention payloads were found in outputs.attentions on the '{split_name}' dataset ({mode_label})."
        )

    layer_sums = state["layer_sums"]
    layer_counts = state["layer_counts"]
    token_intervals = state["token_intervals"]

    mean_heatmaps = {source: [] for source in ATTN_SOURCES}
    for source in ATTN_SOURCES:
        if len(layer_sums[source]) == 0:
            continue

        mean_heatmaps[source] = [
            (layer_sums[source][i] / layer_counts[source][i]).to(torch.float32)
            for i in range(len(layer_sums[source]))
        ]

    if "decoder" in token_intervals and not token_intervals["decoder"]:
        token_intervals["decoder"] = _default_decoder_token_intervals(model)
    if "encoder_fusion" in token_intervals and not token_intervals["encoder_fusion"]:
        token_intervals["encoder_fusion"] = dict(token_intervals.get("decoder", {}))

    return state["raw_attention_payloads"], mean_heatmaps, token_intervals


def _apply_rna_mode(
    values_batch: dict[str, torch.Tensor],
    rna_mode: str,
    permutation: torch.Tensor | None,
):
    """Return batch values transformed according to the selected RNA mode."""
    if rna_mode == "rna_present" or "rna" not in values_batch:
        return values_batch

    mode_values_batch = dict(values_batch)
    if rna_mode == "rna_zeroed":
        mode_values_batch["rna"] = torch.zeros_like(values_batch["rna"])
    elif rna_mode == "rna_permuted":
        if permutation is None:
            batch_size = values_batch["rna"].shape[0]
            permutation = torch.randperm(batch_size, device=values_batch["rna"].device)
        mode_values_batch["rna"] = values_batch["rna"][permutation]
    else:
        raise ValueError(f"Unsupported RNA attention mode: '{rna_mode}'")

    return mode_values_batch


def _collect_attention_heatmaps_for_rna_modes(
    model,
    dataloader,
    device: str,
    split_name: str = "test",
    rna_modes: tuple[str, ...] = _RNA_ATTENTION_MODES,
    collect_raw_attention_payloads: bool = False,
    ATTN_SOURCES: tuple[str, ...] = ("encoder_fusion", "decoder"),
):
    """Collect attention heatmaps for one or more RNA modes in one pass.

    For each incoming batch, the function runs model inference for all requested
    RNA modes (`rna_present`, `rna_zeroed`, `rna_permuted`) and maintains
    independent accumulators per mode.
    """
    if len(rna_modes) == 0:
        raise ValueError("At least one RNA mode must be provided.")

    mode_order = tuple(dict.fromkeys(rna_modes))
    unknown_modes = set(mode_order) - set(_RNA_ATTENTION_MODES)
    if unknown_modes:
        raise ValueError(f"Unsupported RNA attention modes: {sorted(unknown_modes)}")

    mode_labels = {
        "rna_present": "RNA=orig",
        "rna_zeroed": "RNA=0",
        "rna_permuted": "RNA=perm",
    }
    if len(mode_order) == 1:
        desc = f"{split_name} inference | {mode_labels[mode_order[0]]}"
    else:
        desc = f"{split_name} inference | " + " / ".join(mode_labels[mode] for mode in mode_order)

    mode_states = {
        mode: _init_attention_state(
            ATTN_SOURCES=ATTN_SOURCES,
            collect_raw_attention_payloads=collect_raw_attention_payloads,
        )
        for mode in mode_order
    }

    model.eval()
    with torch.no_grad():
        for values_batch, masks_batch in tqdm.tqdm(dataloader, desc=desc):
            values_batch = {modality: value.to(device) for modality, value in values_batch.items()}
            masks_batch = {modality: value.to(device) for modality, value in masks_batch.items()}

            permutation = None
            if "rna_permuted" in mode_order and "rna" in values_batch:
                batch_size = values_batch["rna"].shape[0]
                permutation = torch.randperm(batch_size, device=values_batch["rna"].device)

            for mode in mode_order:
                mode_values_batch = _apply_rna_mode(
                    values_batch=values_batch,
                    rna_mode=mode,
                    permutation=permutation,
                )
                outputs = model(mode_values_batch, masks_batch)
                payload = _normalize_attention_payload(outputs.attentions)
                _accumulate_attention_payload(
                    payload=payload,
                    state=mode_states[mode],
                    ATTN_SOURCES=ATTN_SOURCES,
                )

    results = {}
    for mode in mode_order:
        raw_attention_payloads, mean_heatmaps, token_intervals = _finalize_attention_state(
            state=mode_states[mode],
            model=model,
            split_name=split_name,
            mode_label=mode_labels[mode],
            ATTN_SOURCES=ATTN_SOURCES,
        )
        results[mode] = {
            "raw_attention_payloads": raw_attention_payloads,
            "mean_heatmaps": mean_heatmaps,
            "token_intervals": token_intervals,
        }

    return results


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
    collect_raw_attention_payloads: bool = False,
    ATTN_SOURCES : tuple[str, ...] = ("encoder_fusion", "decoder"),
):
    if zero_rna and permute_rna:
        raise ValueError("Choose only one RNA perturbation mode: zero_rna or permute_rna.")

    if zero_rna:
        rna_mode = "rna_zeroed"
    elif permute_rna:
        rna_mode = "rna_permuted"
    else:
        rna_mode = "rna_present"

    mode_results = _collect_attention_heatmaps_for_rna_modes(
        model=model,
        dataloader=dataloader,
        device=device,
        split_name=split_name,
        rna_modes=(rna_mode,),
        collect_raw_attention_payloads=collect_raw_attention_payloads,
        ATTN_SOURCES=ATTN_SOURCES,
    )
    selected_mode_results = mode_results[rna_mode]
    return (
        selected_mode_results["raw_attention_payloads"],
        selected_mode_results["mean_heatmaps"],
        selected_mode_results["token_intervals"],
    )

##################################################################
# plotting
##################################################################

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


def _resolve_token_location(
    token_index: int,
    intervals: dict[str, tuple[int, int]],
    modality_order: tuple[str, ...],
) -> tuple[str | None, int | None]:
    """Map a global token index to modality and modality-local index."""
    for modality in modality_order:
        if modality not in intervals:
            continue

        start, end = intervals[modality]
        start = int(start)
        end = int(end)
        if start <= token_index < end:
            return modality, token_index - start

    return None, None


def _build_top_key_rows(
    tick_positions: list[int],
    tick_labels: list[str],
    attention_scores: np.ndarray,
    intervals: dict[str, tuple[int, int]],
    modality_order: tuple[str, ...],
    modality_names: dict[str, str],
    token_index_to_description: dict[int, str] | None,
    delta_scores: np.ndarray | None = None,
) -> list[dict[str, int | float | str | None]]:
    """Build rows for top-key token tables used in attention visual analysis."""
    token_index_to_description = token_index_to_description or {}
    rows: list[dict[str, int | float | str | None]] = []

    for token_idx, token_label in zip(tick_positions, tick_labels):
        token_idx = int(token_idx)
        if token_idx < 0 or token_idx >= attention_scores.shape[0]:
            continue

        modality, local_token_index = _resolve_token_location(
            token_index=token_idx,
            intervals=intervals,
            modality_order=modality_order,
        )
        modality_name = "Unknown" if modality is None else modality_names.get(modality, modality.upper())
        gene_cluster_name = token_index_to_description.get(token_idx) or token_label

        row = {
            "token_index": token_idx,
            "modality": modality_name,
            "local_token_index": local_token_index,
            "gene_cluster_name": str(gene_cluster_name),
            "mean_attention_score": float(attention_scores[token_idx]),
        }
        if delta_scores is not None and 0 <= token_idx < delta_scores.shape[0]:
            row["mean_delta_score"] = float(delta_scores[token_idx])
        rows.append(row)

    rows.sort(key=lambda item: (-float(item["mean_attention_score"]), int(item["token_index"])))
    return rows


def _plot_attention_source_comparison(
    source_name: str,
    mean_heatmaps_present,
    mean_heatmaps_zeroed=None,
    mean_heatmaps_permuted=None,
    intervals=None,
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
    on_layer_rendered=None,
):
    mean_heatmaps_present = _as_layer_tuple(mean_heatmaps_present)
    mean_heatmaps_zeroed = _as_layer_tuple(mean_heatmaps_zeroed)
    mean_heatmaps_permuted = _as_layer_tuple(mean_heatmaps_permuted)
    intervals = intervals or {}

    if len(mean_heatmaps_present) == 0 and len(mean_heatmaps_zeroed) == 0 and len(mean_heatmaps_permuted) == 0:
        print(f"No attentions found for source='{source_name}'.")
        return []

    if len(mean_heatmaps_present) == 0:
        raise RuntimeError(
            f"RNA-present heatmaps are required for source='{source_name}' to build comparison plots."
        )

    has_zeroed = len(mean_heatmaps_zeroed) > 0
    has_permuted = len(mean_heatmaps_permuted) > 0

    if has_zeroed and len(mean_heatmaps_present) != len(mean_heatmaps_zeroed):
        raise RuntimeError(
            f"Different number of {source_name} layers between RNA-present and RNA-zeroed runs: "
            f"{len(mean_heatmaps_present)} vs {len(mean_heatmaps_zeroed)}"
        )
    if has_permuted and len(mean_heatmaps_present) != len(mean_heatmaps_permuted):
        raise RuntimeError(
            f"Different number of {source_name} layers between RNA-present and RNA-permuted runs: "
            f"{len(mean_heatmaps_present)} vs {len(mean_heatmaps_permuted)}"
        )

    num_layers = len(mean_heatmaps_present)
    deltas_zeroed = (
        [mean_heatmaps_zeroed[i] - mean_heatmaps_present[i] for i in range(num_layers)]
        if has_zeroed
        else []
    )
    deltas_permuted = (
        [mean_heatmaps_permuted[i] - mean_heatmaps_present[i] for i in range(num_layers)]
        if has_permuted
        else []
    )

    heatmaps_by_name = {
        "present": mean_heatmaps_present,
    }
    if has_zeroed:
        heatmaps_by_name["zeroed"] = mean_heatmaps_zeroed
        heatmaps_by_name["zeroed_minus_present"] = deltas_zeroed
    if has_permuted:
        heatmaps_by_name["permuted"] = mean_heatmaps_permuted
        heatmaps_by_name["permuted_minus_present"] = deltas_permuted

    panel_specs = [
        ("RNA present", "present", "present", None, "main"),
    ]
    if has_zeroed:
        panel_specs.extend(
            [
                ("RNA zeroed", "zeroed", "zeroed", None, "main"),
                ("Zeroed - Present", "zeroed_minus_present", "zeroed", "zeroed_minus_present", "delta"),
            ]
        )
    if has_permuted:
        panel_specs.extend(
            [
                ("RNA permuted", "permuted", "permuted", None, "main"),
                ("Permuted - Present", "permuted_minus_present", "permuted", "permuted_minus_present", "delta"),
            ]
        )

    common_values = torch.cat(
        [
            x.flatten()
            for x in (
                list(mean_heatmaps_present)
                + list(mean_heatmaps_zeroed)
                + list(mean_heatmaps_permuted)
            )
        ]
    ).detach().cpu().numpy()
    common_vmin, common_vmax = _percentile_bounds(common_values, low_q=1.0, high_q=99.0)

    delta_vmin = None
    delta_vmax = None
    if len(deltas_zeroed) > 0 or len(deltas_permuted) > 0:
        delta_values = torch.cat([d.flatten() for d in (deltas_zeroed + deltas_permuted)]).detach().cpu().numpy()
        delta_vmin, delta_vmax = _percentile_bounds(delta_values, low_q=1.0, high_q=99.0)
        if delta_vmax <= 0:
            delta_vmax = 1e-12
        if delta_vmin >= 0:
            delta_vmin = -1e-12

    interval_text = _format_modality_ranges(intervals, MODALITY_ORDER=MODALITY_ORDER, MODALITY_NAMES=MODALITY_NAMES)
    per_layer_top_key_rows = []
    for layer_idx in range(num_layers):
        num_panels = len(panel_specs)
        fig, axes = plt.subplots(1, num_panels, figsize=(max(6, 4 * num_panels), 7))
        if num_panels == 1:
            axes = np.array([axes])

        fig.subplots_adjust(left=0.07, right=0.80, bottom=0.05, top=0.90, wspace=0.18)
        fig.suptitle(
            f"{source_name} attentions | Layer {layer_idx + 1}\nToken ranges: {interval_text}",
            fontsize=13,
            # y=0.96,
        )

        heat_by_name = {
            heatmap_name: heatmap_layers[layer_idx].detach().cpu().numpy()
            for heatmap_name, heatmap_layers in heatmaps_by_name.items()
        }
        heat_present = heat_by_name["present"]

        matrix_size = heat_present.shape[0]
        top_ticks_by_name: dict[str, tuple[list[int], list[str]]] = {}
        key_scores_by_name = {
            "present": heat_present.mean(axis=0),
        }
        if has_zeroed:
            key_scores_by_name["zeroed"] = heat_by_name["zeroed"].mean(axis=0)
        if has_permuted:
            key_scores_by_name["permuted"] = heat_by_name["permuted"].mean(axis=0)

        delta_key_scores_by_name = {}
        if has_zeroed:
            delta_key_scores_by_name["zeroed_minus_present"] = heat_by_name["zeroed_minus_present"].mean(axis=0)
        if has_permuted:
            delta_key_scores_by_name["permuted_minus_present"] = heat_by_name["permuted_minus_present"].mean(axis=0)

        for heatmap_name, key_scores in key_scores_by_name.items():
            top_ticks_by_name[heatmap_name] = _get_top_key_token_ticks(
                key_scores=key_scores,
                intervals=intervals,
                token_index_to_description=token_index_to_description,
                modality_order=MODALITY_ORDER,
                modality_names=MODALITY_NAMES,
                top_k_per_modality=top_k_per_modality,
            )

        layer_tables_by_heatmap = {}
        heatmap_table_specs = [
            (table_name, attention_key, delta_key)
            for table_name, _, attention_key, delta_key, _ in panel_specs
        ]
        for table_name, attention_key, delta_key in heatmap_table_specs:
            tick_positions, tick_labels = top_ticks_by_name.get(attention_key, ([], []))
            layer_tables_by_heatmap[table_name] = _build_top_key_rows(
                tick_positions=tick_positions,
                tick_labels=tick_labels,
                attention_scores=key_scores_by_name[attention_key],
                intervals=intervals,
                modality_order=MODALITY_ORDER,
                modality_names=MODALITY_NAMES,
                token_index_to_description=token_index_to_description,
                delta_scores=(
                    delta_key_scores_by_name.get(delta_key)
                    if delta_key is not None
                    else None
                ),
            )
        per_layer_top_key_rows.append(
            {
                "layer_index": layer_idx + 1,
                "tables_by_heatmap": layer_tables_by_heatmap,
            }
        )

        main_mappable = None
        delta_mappable = None
        main_axes = []
        delta_axes = []
        for col_idx, (title, panel_key, attention_key, _, panel_kind) in enumerate(panel_specs):
            ax = axes[col_idx]
            heat = heat_by_name[panel_key]

            if panel_kind == "main":
                im = ax.imshow(
                    heat,
                    cmap="viridis",
                    vmin=common_vmin,
                    vmax=common_vmax,
                    aspect="equal",
                )
                if main_mappable is None:
                    main_mappable = im
                main_axes.append(ax)
            else:
                im = ax.imshow(
                    heat,
                    cmap="coolwarm",
                    vmin=delta_vmin,
                    vmax=delta_vmax,
                    aspect="equal",
                )
                if delta_mappable is None:
                    delta_mappable = im
                delta_axes.append(ax)

            ax.set_title(title, pad=8)
            ax.set_ylabel("Query tokens")
            ax.set_xlabel("Key tokens")
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
            top_tick_positions, top_tick_labels = top_ticks_by_name.get(attention_key, ([], []))
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

        colorbar_bottom = min(ax.get_position().y0 for ax in axes)
        colorbar_top = max(ax.get_position().y1 for ax in axes)
        colorbar_height = colorbar_top - colorbar_bottom
        colorbar_width = 0.012
        colorbar_gap = 0.090
        colorbar_left = 0.84

        if main_mappable is not None and len(main_axes) > 0:
            main_cax = fig.add_axes(
                [colorbar_left, colorbar_bottom, colorbar_width, colorbar_height]
            )
            cb_main = fig.colorbar(main_mappable, cax=main_cax)
            cb_main.set_label("Attention score")
            colorbar_left += colorbar_width + colorbar_gap
        if delta_mappable is not None and len(delta_axes) > 0:
            delta_cax = fig.add_axes(
                [colorbar_left, colorbar_bottom, colorbar_width, colorbar_height]
            )
            cb_delta = fig.colorbar(delta_mappable, cax=delta_cax)
            cb_delta.set_label("Delta")

        plt.show()
        if on_layer_rendered is not None:
            on_layer_rendered(layer_idx + 1, layer_tables_by_heatmap)

    return per_layer_top_key_rows
