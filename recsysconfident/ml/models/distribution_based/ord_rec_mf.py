import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_ordrec_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = OrdRec(
        num_users=info.n_users,
        num_items=info.n_items,
        num_factors=512,
        rmax=info.rate_range[1],
        rmin=info.rate_range[0],
        items_per_user=info.items_per_user
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class OrdRec(TorchModel):
    def __init__(self, num_users, num_items, num_factors, items_per_user, rmax: float, rmin: float):
        super(OrdRec, self).__init__(items_per_user)
        self.num_ratings = int((rmax - rmin) + 1)
        self.num_users = num_users
        self.num_items = num_items

        self.rmax = rmax
        self.rmin = rmin

        # SVD++ parameters
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.item_bias = nn.Embedding(num_items, 1)
        self.y_j = nn.Embedding(num_items, num_factors)  # implicit feedback

        # Ordinal regression parameters
        self.t1 = nn.Embedding(num_users, 1)  # t_1^u for each user
        self.beta = nn.Embedding(num_users, self.num_ratings - 2)  # β_1^u to β_{S-2}^u

        # Ranking weight vector (w in the paper)
        self.ranking_weights = nn.Parameter(torch.ones(self.num_ratings))

        # Initialize parameters
        self._init_weights()
        self.switch_to_rating()

    def switch_to_ranking(self):
        self.loss = self.fit_ranking_loss
        self.eval_loss = self.ord_rec_loss
        self.predict = self.predict_rank_scores
        self.freeze_all_except_ranking()

    def switch_to_rating(self):
        self.loss = self.fit_prob_loss
        self.eval_loss = self.prob_loss
        self.predict = self.predict_rating
        self.freeze_ranking_weights()

    def _init_weights(self):
        # Initialize SVD++ parameters
        nn.init.xavier_uniform(self.P.weight)
        nn.init.xavier_uniform(self.Q.weight)
        nn.init.xavier_uniform(self.item_bias.weight)
        nn.init.xavier_uniform(self.y_j.weight)

        # Initialize ordinal parameters
        nn.init.xavier_uniform(self.t1.weight)
        nn.init.xavier_uniform(self.beta.weight)

        # Initialize ranking weights
        nn.init.ones_(self.ranking_weights)

    def forward(self, user_id, item_id):
        """Compute the internal score y_ui (Eq. 3 in the paper)"""
        return self._forward(user_id, item_id)

    def _forward(self, user_id, item_id):

        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_i = self.item_bias(item_id).squeeze()

        # Incorporate implicit feedback (Eq. 3)
        user_items_tensor = [torch.tensor(list(self.items_per_user[u.item()][0]),
                                          dtype=torch.long, device=user_id.device)
                             for u in user_id]

        sum_yj = torch.stack([self.y_j(items).sum(dim=0) for items in user_items_tensor])
        norm_factors = torch.tensor([len(items) for items in user_items_tensor],
                                    device=user_id.device).float().sqrt().unsqueeze(1)
        implicit_feedback = sum_yj / norm_factors

        y_ui = b_i + (Q_i * (P_u + implicit_feedback)).sum(dim=1)

        return y_ui

    def get_user_thresholds(self, user_id):
        """Compute ordered thresholds for each user using Eq. 5 from the paper"""
        t1 = self.t1(user_id)  # shape: (batch_size, 1)
        betas = self.beta(user_id)  # shape: (batch_size, num_ratings-2)

        # Compute thresholds t2, t3, ..., t_{S-1} using t_{r+1} = t_r + exp(β_r)
        thresholds = [t1]
        for i in range(self.num_ratings - 2):
            next_threshold = thresholds[-1] + torch.exp(betas[:, i:i + 1]) #ensures t_r-1 <= t_r
            thresholds.append(next_threshold)

        return torch.cat(thresholds, dim=1)

    def predict_proba(self, user_ids, item_ids):
        """Predict the full probability distribution over ratings (Eq. 7-10)"""
        y_ui = self._forward(user_ids, item_ids)
        thresholds = self.get_user_thresholds(user_ids)

        cum_probs = torch.sigmoid(y_ui.unsqueeze(1) - thresholds) #approximated CDF by sigmoid.
        padded_cum_probs = torch.cat([
            torch.zeros_like(y_ui.unsqueeze(1)),
            cum_probs,
            torch.ones_like(y_ui.unsqueeze(1))
        ], dim=1) #shape(batch_size, 1 + n_thresholds + 1)

        probs = padded_cum_probs[:, 1:] - padded_cum_probs[:, :-1]
        return probs

    def fit_ranking_loss(self, user_id, item_id, rating, optimizer):

        optimizer.zero_grad()
        loss = self.ord_rec_loss(user_id, item_id, rating)
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
        probs_i = self.predict_proba(user_id, item_id)  # P(r_ui)
        probs_j = self.predict_proba(user_id, j_ids)  # P(r_uj)

        # Compute ΔP_u(i,j) = P(r_ui) - P(r_uj)
        delta_p = probs_i - probs_j  # shape: (batch_size, num_ratings)

        # Compute w^T ΔP_u(i,j) using learned weights
        scores = torch.matmul(delta_p, self.ranking_weights)  # shape: (batch_size,)

        # Compute logistic loss: -log(σ(score)) = softplus(-score)
        losses = F.softplus(-scores).mean()

        return losses

    def prob_loss(self, user_id, item_id, rating):
        rating_idx = (rating - self.rmin).long()
        probs = self.predict_proba(user_id, item_id)
        true_probs = probs.gather(1, rating_idx.unsqueeze(1)).squeeze()
        true_probs = torch.clamp(true_probs, 1e-10, 1.0)
        nll_loss = -torch.log(true_probs).mean()
        return nll_loss

    def fit_prob_loss(self, user_id, item_id, rating, optimizer):
        optimizer.zero_grad()

        total_loss = self.prob_loss(user_id, item_id, rating) + self.regularization() * 0.0001

        total_loss.backward()
        optimizer.step()

        return total_loss

    def regularization(self):
        """L2 regularization for all parameters"""
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2) ** 2
        return l2_reg

    def predict_rank_scores(self, user_ids, item_ids):
        probs = self.predict_proba(user_ids, item_ids)
        scores = probs @ self.ranking_weights
        p_max = probs.argmax(dim=1)
        confidence = probs.gather(1, p_max.unsqueeze(1)).squeeze(1)  # Shape: [10000]
        return scores, confidence

    def predict_rating(self, user_ids, item_ids):
        """Predict the most likely rating"""
        probs = self.predict_proba(user_ids, item_ids)  # Shape: [10000, 5]
        p_max = probs.argmax(dim=1).unsqueeze(1)  # Shape: [10000]
        ratings = p_max + self.rmin  # Shape: [10000]
        confidence = probs.gather(1, p_max).squeeze(1)  # Shape: [10000]
        return ratings, confidence

    def freeze_ranking_weights(self):
        """Freeze everything except ranking weights"""
        for param in self.parameters():
            param.requires_grad = False
        self.ranking_weights.requires_grad = True

    def freeze_all_except_ranking(self):
        """Freeze ranking weights only"""
        for param in self.parameters():
            param.requires_grad = True
        self.ranking_weights.requires_grad = False
