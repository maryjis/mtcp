import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pycox.evaluation import EvalSurv
import numpy as np
from typing import List

from sksurv.metrics import concordance_index_censored

def create_nan_dataframe(
    num_row: int, num_col: int, name_col: List[str]
) -> pd.DataFrame:
    df = pd.DataFrame(np.zeros((num_row, num_col)), columns=name_col)
    df[:] = np.nan
    return df

def interpolate_dataframe(dataframe: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    dataframe.reset_index(inplace=True)
    dataframe_list = []
    for i, idx in enumerate(dataframe.index):
        df_temp = dataframe[dataframe.index == idx]
        dataframe_list.append(df_temp)
        if i != len(dataframe) - 1:
            dataframe_list.append(
                create_nan_dataframe(n, df_temp.shape[1], df_temp.columns)
            )

    dataframe = pd.concat(dataframe_list).interpolate("linear")
    dataframe = dataframe.set_index("index")
    return dataframe


def compute_survival_metrics(outputs, time, event, cuts=None):
    """
    Compute the survival metrics with the PyCox package.
    
    Parameters
    ----------
    outputs : dict or list of torch.Tensor
        Если dict: ключи - модальности, значения - output logits по времени [N, T]
        Если list: обычный список output logits
    time : torch.Tensor
        Время до события
    event : torch.Tensor
        Флаги событий
    cuts : optional
        Не используется в этой версии
    """
    
    # Если outputs словарь, усредняем risk по модальностям
    if isinstance(outputs, dict):
        risk_list = []
        survival_list = []
        for modality, out in outputs.items():
            out = torch.cat(out, dim=0)
            haz = out.sigmoid()
            cum_h = -torch.log1p(-haz + 1e-7).cumsum(1)
            risk_list.append(cum_h[:, -1].clone())
            surv = (1 - haz).add(1e-7).log().cumsum(1).exp().cpu().numpy()
            survival_list.append(surv)
        
        # Усреднение риска по модальностям
        risk = torch.stack(risk_list, dim=0).mean(dim=0).cpu().numpy()
        
        # Усреднение survival по модальностям
        survival = np.mean(np.stack(survival_list, axis=0), axis=0)
        
    else:
        # outputs — list/tensor
        hazard = torch.cat(outputs, dim=0)
        cum_hazard = -torch.log1p(-hazard.sigmoid() + 1e-7).cumsum(1)
        risk = cum_hazard[:, -1].detach().cpu().numpy()
        survival = (1 - hazard.sigmoid()).add(1e-7).log().cumsum(1).exp().detach().cpu().numpy()
    
    survival_df = pd.DataFrame(survival.transpose())

    # TODO check why we use interpolation here!
    # survival_df = interpolate_dataframe(pd.DataFrame(survival.transpose(), cuts))
    
    evaluator = EvalSurv(
        survival_df, time.cpu().numpy(), event.cpu().numpy(), censor_surv="km"
    )
    c_index = evaluator.concordance_td()
    ibs = evaluator.integrated_brier_score(np.linspace(0, time.cpu().numpy().max()))
    inbll = evaluator.integrated_nbll(np.linspace(0, time.cpu().numpy().max()))
    cs_score = (c_index + (1 - ibs)) / 2
    c_index_v2 = concordance_index_censored(event.cpu().numpy().astype(bool),
                                            time.cpu().numpy(),
                                            risk,
                                            tied_tol=1e-08)[0]
    
    return {"c_index": c_index, "ibs": ibs, "inbll": inbll,
            "cs_score": cs_score, "c_index_v2": c_index_v2}
    
def agg_fold_metrics(lst: list[dict[str, float]]):
    """Compute mean, min, max, std from cross validation metrics"""
    keys = lst[0].keys()
    res = {}
    for k in keys:
        res[k] = compute_stats([dct[k] for dct in lst])
    return res


def compute_stats(lst: list[float]) -> dict[str, np.ndarray]:
    """Compute some stats from a list of floats"""
    arr = np.array(lst)
    return {"mean": arr.mean(), "std": arr.std(), "min": arr.min(), "max": arr.max()}