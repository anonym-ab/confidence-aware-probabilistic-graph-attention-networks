"""
package: recsysconfident.ranking.elementwise_error
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.environment import Environment
from recsysconfident.ml.ranking.rank_helper import get_low_rank_items
from recsysconfident.ml.eval.predict_helper import predict
from recsysconfident.ml.models.torchmodel import TorchModel


def elementwise_pos_neg_scores(model, split_df: pd.DataFrame, environ: Environment, device) -> (pd.DataFrame, pd.DataFrame):
    pos_indices = split_df[environ.dataset_info.relevance_col] >= environ.dataset_info.r_t
    positive_split_df = split_df[pos_indices]
    neg_true, neg_pred, neg_conf = obtain_neg_scores(model,
                                                     torch.from_numpy(
                                                         positive_split_df[environ.dataset_info.user_col].values),
                                                     environ.dataset_info,
                                                     device)
    positive_split_df.loc[:, environ.dataset_info.relevance_col] = 1
    positive_split_df.loc[:, "neg_pred"], positive_split_df.loc[:, "neg_conf"] = neg_pred, neg_conf

    neg_split_df = split_df[~pos_indices]
    neg_true, neg_pred, neg_conf = obtain_neg_scores(model,
                                                     torch.from_numpy(
                                                         neg_split_df[environ.dataset_info.user_col].values),
                                                     environ.dataset_info,
                                                     device)
    neg_split_df.loc[:, "neg_pred"], neg_split_df.loc[:, "neg_conf"] = neg_pred, neg_conf
    neg_split_df.loc[:, environ.dataset_info.relevance_col] = 0
    split_df = pd.concat([neg_split_df, positive_split_df], axis=1)
    return split_df


def elementwise_abs_loss(model: TorchModel, split_df, dataloader, environ, device):

    y_true, y_pred, conf_pred = predict(model, dataloader, device)

    test_neg_true, test_neg_pred, test_neg_conf = obtain_neg_scores(model,
                                                                    torch.from_numpy(split_df[environ.dataset_info.user_col].values),
                                                                    environ.dataset_info,
                                                                    device)

    split_df.loc[:, environ.dataset_info.r_pred_col], split_df.loc[:, environ.dataset_info.conf_pred_col] = y_pred, conf_pred
    split_df.loc[:, "neg_pred"], split_df.loc[:, "neg_conf"] = test_neg_pred, test_neg_conf

    diff_sum = (split_df[environ.dataset_info.relevance_col] - y_true).sum()
    assert (diff_sum == 0, f"The datasets are not synchronized: {diff_sum}")
    print("Exporting test split predictions")

    set_bpr_error(split_df)
    return split_df

def set_bpr_error(df: pd.DataFrame):
    diff = df['r_pred'] - df['neg_pred']
    df.loc[:, "bpr_error"] = -torch.log(torch.sigmoid(torch.from_numpy(diff.values)) + 1e-8)
    return df

def obtain_neg_scores(model: TorchModel, users_ids: torch.Tensor, data_info:DatasetInfo, device):
    """
    Obtain negative items scores for evaluation during training only.: For each user, sample only one negative item.
    """
    test_low_rank_items = get_low_rank_items(users_ids,
                                             data_info.items_per_user,
                                             data_info.n_items)

    neg_items_dataloader = DataLoader(
        TensorDataset(users_ids,
                      test_low_rank_items,
                      torch.zeros_like(users_ids)),
        batch_size=1024,
        shuffle=False)
    return predict(model, neg_items_dataloader, device)
