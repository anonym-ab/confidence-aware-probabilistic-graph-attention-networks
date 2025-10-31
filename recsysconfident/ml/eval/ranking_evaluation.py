"""
package: recsysconfident.ml.fit_eval
"""
import pandas as pd
from pandas import DataFrame

from recsysconfident.constants import NEG_FLAG_COL
from recsysconfident.environment import Environment
from recsysconfident.ml.distance_metrics import rmse
from recsysconfident.ml.ranking.rank_metrics import ConfAwareRankingMetrics


def ranking_scores(candidates_norm_df: DataFrame, environ: Environment, k=10) -> dict:

    conf_rank_calculator = ConfAwareRankingMetrics(environ.dataset_info)
    rank_scores_mean, rank_scores_std = conf_rank_calculator.users_mean_std_rank_metrics(candidates_norm_df, k)

    scores_dict = {
        f"mNDCG@{k}": f"{rank_scores_mean[0]:.5f}",
        f"stdNDCG@{k}": f"{rank_scores_std[0]:.5f}",
        f"MAP@{k}": f"{rank_scores_mean[1]:.5f}",
        f"stdMAP@{k}": f"{rank_scores_std[1]:.5f}",
    }
    return scores_dict

def evaluate(split_df: pd.DataFrame, environ: Environment) -> dict:

    distance_metrics = get_distance_metrics(split_df, environ)
    rmax = environ.dataset_info.rate_range[1]
    rmin = environ.dataset_info.rate_range[0]

    split_df.loc[:, environ.dataset_info.relevance_col] = (split_df[environ.dataset_info.relevance_col] - rmin) / (rmax - rmin)
    split_df.loc[:, environ.dataset_info.r_pred_col] = (split_df[environ.dataset_info.r_pred_col] - rmin) / (rmax - rmin)

    ranking_10metrics = ranking_scores(split_df, environ, 10)
    ranking_3metrics = ranking_scores(split_df, environ,  3)

    return {**distance_metrics, **ranking_10metrics, **ranking_3metrics}

def get_distance_metrics(split_df: pd.DataFrame, environ: Environment):

    non_negative_sampled_df = split_df[split_df[NEG_FLAG_COL] == 0] #We don't actually know the true score for the negative samples since they are non-observed items.
    y_true = non_negative_sampled_df[environ.dataset_info.relevance_col].values
    y_pred = non_negative_sampled_df[environ.dataset_info.r_pred_col].values
    #mae_score = mae(y_true, y_pred)
    rmse_score = rmse(y_true, y_pred)

    return {
        "rmse": rmse_score,
    }

