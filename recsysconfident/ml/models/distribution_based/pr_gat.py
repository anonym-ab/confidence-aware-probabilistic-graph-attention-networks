import torch
import torch.nn as nn
import torch.distributions as d
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_prgat_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = PRGAT(
        n_users=info.n_users,
        n_items=info.n_items,
        emb_dim=64,
        rate_range=info.rate_range
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class PRGAT(TorchModel):

    def __init__(self, n_users, n_items, emb_dim, rate_range: list):
        super().__init__(None)

        self.n_users = n_users
        self.rmin = rate_range[0]
        self.rmax = rate_range[1]
        self.delta_r = rate_range[2]/2

        self.ui_lookup = nn.Embedding(n_users + n_items, emb_dim)

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                   out_channels=emb_dim,
                                   heads=1,
                                   concat=False
                                   )
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

        # Variance parameters (γ_u, γ_v), initialized to 1.0
        self.user_gamma = nn.Embedding(n_users, 1)
        self.item_gamma = nn.Embedding(n_items, 1)
        nn.init.ones_(self.user_gamma.weight)
        nn.init.ones_(self.item_gamma.weight)

        self.alpha = nn.Parameter(torch.tensor(1.))
        nn.init.xavier_uniform_(self.ui_lookup.weight)
        self.mse_loss = nn.MSELoss()
        self.switch_to_rating()

    def forward(self, users_ids, items_ids):

        ui_edges = torch.stack([users_ids, items_ids + self.n_users]) #(batch,),(batch,) -> (2, batch)

        ui_x = self.ui_lookup.weight
        ui_graph_emb = self.ui_gat_layer(x=ui_x, edge_index=ui_edges)  # (max_u_id+1, emb_dim)

        u_graph_emb = ui_graph_emb[ui_edges[0]]
        i_graph_emb = ui_graph_emb[ui_edges[1]]

        x = F.leaky_relu(self.fc1(torch.concat([u_graph_emb, i_graph_emb], dim=1)))
        x = self.dropout(x)
        mean = self.fc2(x).squeeze()

        # Softplus ensures γ > 0
        gamma_u = torch.clamp(self.user_gamma(users_ids), min=0.00001) #the article does not mention, but it does not work without.
        gamma_v = torch.clamp(self.item_gamma(items_ids), min=0.00001)
        alpha = torch.exp(self.alpha)

        precision = alpha * gamma_u * gamma_v
        variance = 1.0 / precision
        std = torch.sqrt(variance).squeeze() #=> precision = 1/(std * std) = 1/var

        return torch.stack([mean, std], dim=1)

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()
        pred_scores = self.forward(user_ids, item_ids)
        true_scores_norm = (labels - self.rmin) / (self.rmax - self.rmin)
        mu = pred_scores[:, 0]
        sigma = pred_scores[:, 1]
        nll = -d.Normal(mu, sigma).log_prob(true_scores_norm).mean()
        mse_loss = self.mse_loss(mu, labels)

        loss = nll + mse_loss * 0.001
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, user_ids, item_ids, labels):
        pred_scores = self.forward(user_ids, item_ids)
        mu = pred_scores[:, 0]
        rating = mu * (self.rmax - self.rmin) + self.rmin
        return torch.sqrt(torch.nn.functional.mse_loss(labels, rating, reduction='mean'))

    def regularization(self):
        return 0

    def sharpe_ratio(self, R, sigma):
        return (R - 0.7 * self.rmax) / (sigma + 0.00001)

    def raking_predict(self, user_ids, item_ids):

        scores = self.forward(user_ids, item_ids)

        mu = scores[:, 0]
        sigma = scores[:, 1]
        dist = d.Normal(scores[:, 0], sigma)

        R = mu * (self.rmax - self.rmin) + self.rmin
        confidence = dist.cdf(mu + self.delta_r) - dist.cdf(mu - self.delta_r)
        score = self.sharpe_ratio(R, sigma)

        return score, confidence

    def rating_predict(self, user_ids, item_ids):
        scores = self.forward(user_ids, item_ids)

        mu = scores[:, 0]
        sigma = scores[:, 1]
        dist = d.Normal(scores[:, 0], sigma)

        R = mu * (self.rmax - self.rmin) + self.rmin
        confidence = dist.cdf(mu + self.delta_r) - dist.cdf(mu - self.delta_r)

        return R, confidence

    def switch_to_ranking(self):
        self.predict = self.raking_predict

    def switch_to_rating(self):
        self.predict = self.rating_predict
