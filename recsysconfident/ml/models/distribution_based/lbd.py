import torch
import torch.nn as nn
from torch.distributions import Beta

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.ml.models.torchmodel import TorchModel


def beta_cdf(x_batch, alpha, beta, npts=10000, eps=1e-7):
    x_batch = x_batch.unsqueeze(-1)  # (batch, 1)
    alpha = alpha.unsqueeze(-1)  # (batch, 1)
    beta = beta.unsqueeze(-1)  # (batch, 1)
    x = torch.linspace(0, 1, npts, device=x_batch.device)  # (npts,)
    x = x.unsqueeze(0) * x_batch  # (batch, npts)

    # Compute PDF (use log-exp for numerical stability)
    log_pdf = Beta(alpha, beta).log_prob(x.clamp(eps, 1 - eps))
    pdf = torch.exp(log_pdf)

    return torch.trapz(pdf, x, dim=-1)

def get_lbd_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = LBD(
        num_users=info.n_users,
        num_items=info.n_items,
        num_hidden=522,
        rmax=info.rate_range[1],
        rmin=info.rate_range[0]
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class LBD(TorchModel):

    def __init__(self, num_users: int, num_items: int, num_hidden: int, rmax: float = 5.0, rmin: float = 0.0):
        super().__init__(None)

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden = num_hidden
        self.rmax = torch.scalar_tensor(rmax)
        self.rmin = torch.scalar_tensor(rmin)

        self.uid_features = nn.Embedding(num_users + 1, num_hidden)
        self.iid_features = nn.Embedding(num_items + 1, num_hidden)
        self.u_bias = nn.Embedding(num_users + num_items + 1, 1)
        self.nu_bias = nn.Embedding(num_users + num_items + 1, 1)
        self.a = nn.Embedding(num_users + num_items + 1, 1)
        self.b = nn.Embedding(num_users + num_items + 1, 1)

        self.u_0 = nn.Parameter(torch.tensor(0.5))
        self.v_0 = nn.Parameter(torch.tensor(0.2))
        self.a_0 = nn.Parameter(torch.tensor(0.1))
        self.b_0 = nn.Parameter(torch.tensor(0.3))

        self.epslon = torch.scalar_tensor(0.0001)
        self.initialize_weights()

    def initialize_weights(self):

        nn.init.xavier_uniform_(self.uid_features.weight)
        nn.init.xavier_uniform_(self.iid_features.weight)
        nn.init.xavier_uniform_(self.u_bias.weight)
        nn.init.xavier_uniform_(self.nu_bias.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)

    def forward(self, u_ids, i_ids):
        U = self.uid_features(u_ids)
        V = self.iid_features(i_ids)

        u_i = self.u_bias(u_ids).squeeze(-1)  # (batch_size,)
        u_j = self.u_bias(i_ids + self.num_users).squeeze(-1)  # (batch_size,)
        threshold = self.u_0 * u_i * u_j

        mu = 0.5 * (1 + nn.functional.cosine_similarity(U, V, dim=-1))

        mu_prime = torch.where(
            mu >= threshold,
            mu / (2 * threshold),
            0.5 + 0.5 * (mu - threshold) / ((2 * (1 - threshold)) + self.epslon)
        )
        mu_prime = torch.clamp(mu_prime, 0.0, 1.0)

        nu = torch.norm(U + V, dim=1)  # shape: (B,)

        nu_i = self.nu_bias(u_ids).squeeze(-1)  # shape: (B,)
        nu_j = self.nu_bias(i_ids + self.num_users).squeeze(-1)  # shape: (B,)
        nu_prime = self.v_0 * nu_i * nu_j * nu  # shape: (B,)

        alpha = nu_prime * mu_prime  # shape: (B,)
        beta = nu_prime * (1 - mu_prime)  # shape: (B,)

        # Squeeze the bias terms to ensure they're (B,) shape
        a_user = self.a(u_ids).squeeze(-1)  # shape: (B,)
        a_item = self.a(i_ids + self.num_users).squeeze(-1)  # shape: (B,)
        b_user = self.b(u_ids).squeeze(-1)  # shape: (B,)
        b_item = self.b(i_ids + self.num_users).squeeze(-1)  # shape: (B,)

        alpha_prime = torch.maximum(self.a_0 + a_user + a_item + alpha, self.epslon)  # shape: (B,)
        beta_prime = torch.maximum(self.b_0 + b_user + b_item + beta, self.epslon)  # shape: (B,)

        return torch.stack([mu_prime, nu_prime, alpha_prime, beta_prime], dim=1)

    def regularization(self):

        U_l2 = torch.norm(self.uid_features.weight, p=2) ** 2
        V_l2 = torch.norm(self.iid_features.weight, p=2) ** 2

        return (U_l2 + V_l2) * 0.0001

    def predict(self, user, item):

        outputs = self.forward(user, item)
        mu, nu = outputs[:, 0], outputs[:, 1]
        ratings = mu * (self.rmax - self.rmin) + self.rmin
        return ratings, nu

    def eval_loss(self, user_ids, item_ids, true_labels):
        model_scores = self.forward(user_ids, item_ids)
        mu = model_scores[:,0]
        ratings = mu * (self.rmax - self.rmin) + self.rmin
        return torch.sqrt(torch.nn.functional.mse_loss(ratings, true_labels, reduction='mean'))

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()

        model_scores = self.forward(user_ids, item_ids)
        alpha = model_scores[:, 2]  # (batch,)
        beta = model_scores[:, 3]  # (batch,)
        ratings_norm = (labels - self.rmin) / (self.rmax - self.rmin) #(batch,)
        n_bins = int((self.rmax - self.rmin) * 2) + 1
        bin_edges = torch.linspace(0, 1, steps=n_bins+1,
                                   device=alpha.device).expand(len(alpha), -1) #(batch_size, n_bins)
        bins_cdf = beta_cdf(bin_edges,
                            alpha.unsqueeze(1),
                            beta.unsqueeze(1))

        # since we have n_bins, which position does the current rating (label) belong to?
        bin_probs = bins_cdf[:, 1:] - bins_cdf[:, :-1]
        bin_indices = (ratings_norm * (bin_probs.size(1) - 1)).long().clamp(0, bin_probs.size(1) - 1)
        true_bin_probs = bin_probs.gather(1, bin_indices.unsqueeze(1)) #get the predicted probability for the true bin position.
        true_bin_probs = torch.clamp(true_bin_probs, min=1e-10, max=1.0)
        loss = -torch.log(true_bin_probs).mean() #The predicted probability for those bins should be 1.

        loss = loss + self.regularization()

        loss.backward()
        optimizer.step()

        return loss
