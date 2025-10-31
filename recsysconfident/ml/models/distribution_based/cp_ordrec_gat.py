import torch
import torch.nn as nn
import torch.distributions as d
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_cpordrecgat_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = CPOrdrecGAT(
        n_users=info.n_users,
        n_items=info.n_items,
        emb_dim=64,
        rmin=info.rate_range[0],
        rmax=info.rate_range[1],
        items_per_user=info.items_per_user
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class CPOrdrecGAT(TorchModel):

    def __init__(self, n_users, n_items, emb_dim, rmin: float, rmax: float, items_per_user: dict):
        super().__init__(items_per_user)

        self.n_users = n_users
        self.rmin = rmin
        self.rmax = rmax
        n_bins = int(2 * (rmax - rmin ))
        self.delta_r = 1 / n_bins

        self.ui_lookup = nn.Embedding(n_users + n_items, emb_dim)

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                   out_channels=emb_dim,
                                   heads=1,
                                   concat=False
                                   )
        nn.init.xavier_uniform(self.ui_lookup.weight)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

        # Variance parameters (γ_u, γ_v), initialized to 1.0
        self.user_gamma = nn.Embedding(n_users, 1)
        self.item_gamma = nn.Embedding(n_items, 1)
        nn.init.ones_(self.user_gamma.weight)
        nn.init.ones_(self.item_gamma.weight)

        self.alpha = nn.Parameter(torch.tensor(1.))
        self.ranking_weights = nn.Parameter(torch.ones(n_bins))

        self.mse_loss = nn.MSELoss()
        self.switch_to_rating()

    def switch_to_ranking(self):
        self.loss = self.ranking_loss
        self.eval_loss = self.ord_rec_loss
        self.predict = self.predict_rank_scores
        self.freeze_all_except_ranking()

    def switch_to_rating(self):
        self.loss = self.prob_rating_loss
        self.eval_loss = self.eval_rmse_loss
        self.predict = self.predict_rating
        self.freeze_ranking_weights()

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

    def prob_rating_loss(self, user_ids, item_ids, labels, optimizer):
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

    def ord_rec_loss(self, user_id, item_id, rating):
        """
        Implements the pairwise ranking loss from Section 4 of the paper.
        For each (u,i) pair, samples a random j from user's rated items and computes:
        - ΔP_u(i,j) = P(r_ui = r | Θ) - P(r_uj = r | Θ)
        - Loss = -log(σ(w^T ΔP_u(i,j))) where r_ui > r_uj
        """

        device = user_id.device

        # Get all items rated by each user in batch
        user_items = [list(self.items_per_user[u.item()][0]) for u in user_id]
        user_ratings = [list(self.items_per_user[u.item()][1]) for u in user_id]

        # Sample random j for each user in batch
        j_indices = [random.randrange(len(items)) for items in user_items]
        j_ids = torch.tensor([items[idx] for items, idx in zip(user_items, j_indices)],
                             device=device)
        j_ratings = torch.tensor([ratings[idx] for ratings, idx in zip(user_ratings, j_indices)],
                                 device=device)

        # Only keep pairs where r_ui > r_uj
        mask = rating > j_ratings
        if not mask.any():
            return torch.tensor(0.0, device=device)

        user_id = user_id[mask]
        item_id = item_id[mask]
        j_ids = j_ids[mask]

        # Get probability distributions
        probs_i = self.compute_probs(user_id, item_id)  # P(r_ui)
        probs_j = self.compute_probs(user_id, j_ids)  # P(r_uj)

        # Compute ΔP_u(i,j) = P(r_ui) - P(r_uj)
        delta_p = probs_i - probs_j  # shape: (batch_size, n_bins)

        # Compute w^T ΔP_u(i,j) using learned weights
        scores = torch.matmul(delta_p, self.ranking_weights)  # shape: (batch_size,)

        # Compute logistic loss: -log(σ(score)) = softplus(-score)
        losses = F.softplus(-scores).mean()

        return losses

    def ranking_loss(self, user_id, item_id, rating, optimizer):

        optimizer.zero_grad()
        loss = self.ord_rec_loss(user_id, item_id, rating)
        loss.backward()
        optimizer.step()
        return loss

    def eval_rmse_loss(self, user_ids, item_ids, labels):
        pred_scores = self.forward(user_ids, item_ids)
        mu = pred_scores[:, 0]
        rating = mu * (self.rmax - self.rmin) + self.rmin
        return torch.sqrt(torch.nn.functional.mse_loss(labels, rating, reduction='mean'))

    def regularization(self):
        return 0

    def predict_rating(self, user_ids, item_ids):
        scores = self.forward(user_ids, item_ids)

        mu = scores[:, 0]
        sigma = scores[:, 1]
        dist = d.Normal(scores[:, 0], sigma)

        R = mu * (self.rmax - self.rmin) + self.rmin
        confidence = dist.cdf(mu + self.delta_r) - dist.cdf(mu - self.delta_r)

        return R, confidence

    def compute_probs(self, user_ids, item_ids):
        pred_scores = self.forward(user_ids, item_ids)  # (batch, 2)
        mu = pred_scores[:, 0]  # (batch,)
        sigma = pred_scores[:, 1]  # (batch,)
        nll = d.Normal(mu.unsqueeze(1), sigma.unsqueeze(1))

        # Create bin centers S1, S2, ..., Sn
        S = torch.arange(self.delta_r, 1 + self.delta_r, self.delta_r, device=mu.device)  # (n_bins,)

        # Compute bin edges
        lower = S - self.delta_r  # (n_bins,)
        upper = S + self.delta_r  # (n_bins,)

        # Compute probability mass in each bin
        probs = nll.cdf(upper) - nll.cdf(lower)  # (batch, n_bins)
        return probs

    def predict_rank_scores(self,  user_ids, item_ids):
        probs = self.compute_probs(user_ids, item_ids)
        scores = probs @ self.ranking_weights  # (batch,)
        p_max = probs.argmax(dim=1)  # (batch,)
        confidence = probs.gather(1, p_max.unsqueeze(1)).squeeze(1)  # (batch,)
        return scores, confidence

    def freeze_ranking_weights(self):
        """Freeze everything except ranking weights"""
        for param in self.parameters():
            param.requires_grad = True
        self.ranking_weights.requires_grad = False

    def freeze_all_except_ranking(self):
        """Freeze ranking weights only"""
        for param in self.parameters():
            param.requires_grad = False
        self.ranking_weights.requires_grad = True
