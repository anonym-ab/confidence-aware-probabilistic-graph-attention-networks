"""
package: recsysconfident.ml.ranking.conf_aware_rank_metrics
"""
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score, recall_score

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo


class ConfAwareRankingMetrics:

    def __init__(self, data_info: DatasetInfo, r_t: float=0.75, alpha: float = 5):
        self.data_info = data_info
        self.r_t = r_t
        self.alpha = alpha

    def calibrate_ratings(self, df: pd.DataFrame, c_t: float):
        def calibrate(row):
            r, c = row[self.data_info.r_pred_col], row[self.data_info.conf_pred_col]
            if r >= self.r_t and c >= c_t:
                return r + self.alpha * c
            else:
                return r

        df[self.data_info.r_pred_col] = df.apply(calibrate, axis=1)
        return df

    def binarize(self, true_relevances):
        return (true_relevances >= self.r_t).astype(int)

    def reciprocal_rank_at_k(self, pred_relevances, true_relevance, k):
        top_k_indices = np.argsort(-pred_relevances)[:k]
        for rank, idx in enumerate(top_k_indices, start=1):
            if true_relevance[idx] >= 1:
                return 1.0 / rank
        return 0.0

    def _get_true_pred_scores(self, df: pd.DataFrame) -> dict:

        user_true_pred_scores = df.groupby(self.data_info.user_col).apply(
            lambda x: (
                x.sort_values(by=self.data_info.r_pred_col, ascending=False)[self.data_info.relevance_col].values,
                x.sort_values(by=self.data_info.r_pred_col, ascending=False)[self.data_info.r_pred_col].values
            )
        ).to_dict()

        return user_true_pred_scores

    def rank_metrics(self, df: pd.DataFrame, k: int) -> list:

        user_true_pred_scores = self._get_true_pred_scores(df)
        metrics = []
        for user_key in user_true_pred_scores.keys():
            true_ratings, pred_ratings = user_true_pred_scores[user_key]
            binary_pred = self.binarize(pred_ratings)
            binary_true = self.binarize(true_ratings)

            metrics.append([
                ndcg_score([true_ratings], [pred_ratings], k=k),
                average_precision_score(binary_true[:k], binary_pred[:k]),
                recall_score(binary_true[:k], binary_pred[:k], average="micro"),
                self.reciprocal_rank_at_k(binary_pred, binary_true, k)
            ])
        return metrics

    def users_mean_std_rank_metrics(self, candidates_df: pd.DataFrame, k: int) -> tuple:
        users_scores = self.rank_metrics(candidates_df, k)
        scores = np.array(users_scores)

        mean_metrics = np.mean(scores, axis=0)
        std_metrics = np.std(scores, axis=0)

        return mean_metrics, std_metrics
