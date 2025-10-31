import torch
import torch.nn as nn
from torch.distributions import Beta

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.ml.models.torchmodel import TorchModel

def beta_cdf(x_batch, alpha, beta, npts=10000, eps=1e-7):
    # x_batch: (B, N+1)
    x_batch = x_batch.unsqueeze(-1)                 # (B, N+1, 1)
    alpha = alpha.unsqueeze(-1).unsqueeze(-1)       # (B, 1, 1)
    beta  = beta.unsqueeze(-1).unsqueeze(-1)        # (B, 1, 1)
    x = torch.linspace(0, 1, npts, device=x_batch.device)  # (npts,)
    x = x.unsqueeze(0).unsqueeze(0) * x_batch       # (B, N+1, npts)

    log_pdf = Beta(alpha, beta).log_prob(x.clamp(eps, 1 - eps))  # (B, N+1, npts)
    pdf = torch.exp(log_pdf)
    return torch.trapz(pdf, x, dim=-1)              # (B, N+1)


def get_lbd_model_and_dataloader(info: DatasetInfo):
    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    # If explicit rate_range is not available, default to 10
    if not (info.rate_range is None) and len(info.rate_range) == 3:
        # Assuming Rmax, Rmin, Rstep
        Rmin, Rmax, Rstep = info.rate_range
        # n = (Rmax - Rmin) / Rstep + 1, but for MovieLens, the 0.5 step results in 10 unique values
        n_ratings = int(round((Rmax - Rmin) / Rstep)) + 1 if Rmin == 0.0 else int(round((Rmax - Rmin) / Rstep))
    else:
        n_ratings = 10  # Default for MovieLens 10M (0.5 to 5.0)

    model = LBD(
        num_users=info.n_users,
        num_items=info.n_items,
        num_hidden=512,  # LBD-A used 512 embeddings
        n_ratings=n_ratings,
        rmax=info.rate_range[1],
        rmin=info.rate_range[0]
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class LBD(TorchModel):

    def __init__(self, num_users: int, num_items: int, num_hidden: int, n_ratings: int, rmax: float = 5.0,
                 rmin: float = 0.0):
        super().__init__(None)

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden = num_hidden
        self.n_ratings = n_ratings
        self.rmax = torch.scalar_tensor(rmax)
        self.rmin = torch.scalar_tensor(rmin)
        self.R_step = (rmax - rmin) / (n_ratings - 1) if n_ratings > 1 else 1.0

        # LBD-A uses 512 embeddings
        self.uid_features = nn.Embedding(num_users + 1, num_hidden)
        self.iid_features = nn.Embedding(num_items + 1, num_hidden)

        # LBD-A uses alpha/beta bias terms (Equation 8)
        self.a = nn.Embedding(num_users + num_items + 1, 1)
        self.b = nn.Embedding(num_users + num_items + 1, 1)

        self.a_0 = nn.Parameter(torch.tensor(0.1))  # Global alpha bias (a_0)
        self.b_0 = nn.Parameter(torch.tensor(0.3))  # Global beta bias (b_0)

        # Adaptive binning parameters (theta_i^r and theta_j^r) - Section 4.4
        # Mapped to n_ratings
        # Using separate embeddings for user and item binning parameters (w_ij,r)
        self.bin_a = nn.Embedding(num_users + 1, n_ratings)
        self.bin_b = nn.Embedding(num_items + 1, n_ratings)

        self.epslon = torch.scalar_tensor(0.0001)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all layers
        nn.init.xavier_uniform_(self.uid_features.weight)
        nn.init.xavier_uniform_(self.iid_features.weight)
        nn.init.xavier_uniform_(self.a.weight)
        nn.init.xavier_uniform_(self.b.weight)
        nn.init.xavier_uniform_(self.bin_a.weight)
        nn.init.xavier_uniform_(self.bin_b.weight)

    def forward(self, u_ids, i_ids):
        # 1. MF-like Preference/Confidence calculation
        U = self.uid_features(u_ids)  # (B, H)
        V = self.iid_features(i_ids)  # (B, H)

        # Predicted Mean (mu) - Cosine Similarity (Equation 6)
        # mu in (0, 1)
        mu = 0.5 * (1 + nn.functional.cosine_similarity(U, V, dim=-1))  # (B,)

        # Predicted Confidence (nu) - v^sum (Equation 7)
        # nu in (0, inf)
        nu = torch.norm(U + V, dim=1)  # (B,)

        # 2. Reparameterization: alpha = mu * nu, beta = (1-mu) * nu
        alpha = nu * mu  # (B,)
        beta = nu * (1 - mu)  # (B,)

        # 3. Apply alpha/beta bias (Equation 8)
        # Note: self.num_users is used as offset for item bias indices
        a_user = self.a(u_ids).squeeze(-1)  # (B,)
        a_item = self.a(i_ids + self.num_users).squeeze(-1)  # (B,)
        b_user = self.b(u_ids).squeeze(-1)  # (B,)
        b_item = self.b(i_ids + self.num_users).squeeze(-1)  # (B,)

        # alpha_prime = max(a_0 + a_i + a_j + alpha, epsilon)
        alpha_prime = torch.maximum(self.a_0 + a_user + a_item + alpha, self.epslon)  # (B,)
        beta_prime = torch.maximum(self.b_0 + b_user + b_item + beta, self.epslon)  # (B,)

        # 4. Adaptive Binning Weights (w_ij,r) - Section 4.4
        # Get theta_i^r and theta_j^r
        theta_u = self.bin_a(u_ids)  # (B, n_ratings)
        theta_i = self.bin_b(i_ids)  # (B, n_ratings)

        # w_ij,r = exp(theta_i^r + theta_j^r)
        w_ij_r = torch.exp(theta_u + theta_i)  # (B, n_ratings)

        # Return unbiased mean/confidence, final distribution parameters, and binning weights
        mu_conf = torch.stack([mu, nu, alpha_prime, beta_prime], dim=1)

        return mu_conf, w_ij_r

    def regularization(self):
        # L2 regularization on embeddings U and V (as per initial code)
        U_l2 = torch.norm(self.uid_features.weight, p=2) ** 2
        V_l2 = torch.norm(self.iid_features.weight, p=2) ** 2

        return (U_l2 + V_l2) * 0.0001

    def predict(self, user, item):
        # Outputs: (B, 4) of [mu, nu, alpha_prime, beta_prime] and (B, N) of w_ij_r
        outputs, _ = self.forward(user, item)
        mu, nu = outputs[:, 0], outputs[:, 1]

        # Prediction is the mean of the Beta distribution: E[x'] = E[x](Rmax - Rmin) + Rmin
        ratings = mu * (self.rmax - self.rmin) + self.rmin
        return ratings, nu  # Returns mean prediction and confidence (nu)

    def eval_loss(self, user_ids, item_ids, true_labels):
        # Evaluation uses RMSE on the predicted mean rating (mu)
        outputs, _ = self.forward(user_ids, item_ids)
        mu = outputs[:, 0]
        ratings = mu * (self.rmax - self.rmin) + self.rmin
        return torch.sqrt(torch.nn.functional.mse_loss(ratings, true_labels, reduction='mean'))

    def loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()

        outputs, w_ij_r = self.forward(user_ids, item_ids)
        alpha = outputs[:, 2]  # (B,)
        beta = outputs[:, 3]  # (B,)

        # --- 1. Determine Adaptive Normalized Bin Boundaries (e_r) ---

        # Normalize the adaptive weights W_ij,r
        W_ij_r = w_ij_r / torch.sum(w_ij_r, dim=1, keepdim=True)  # (B, N)

        # Calculate the normalized bin edges e_r. N+1 edges for N bins.
        # e_1 = 0.0, e_N+1 = 1.0. The cumsum gives e_2 to e_N+1.
        zero_edge = torch.zeros_like(W_ij_r[:, :1])  # (B, 1)
        # normalized_edges shape: (B, N+1)
        normalized_edges = torch.cat(
            [zero_edge, torch.cumsum(W_ij_r, dim=1)], dim=1
        )

        # --- 2. Calculate Probability Mass P(R_r) = F_B(e_{r+1}) - F_B(e_r) (Equation 4) ---

        # CDF at all N+1 edges: (B, N+1)
        cdf_at_edges = beta_cdf(normalized_edges, alpha, beta)

        # Bin probabilities P(R_r) for r=1 to N. P(R_r) is the mass between e_r and e_{r+1}
        # bin_probs shape: (B, N)
        bin_probs = cdf_at_edges[:, 1:] - cdf_at_edges[:, :-1]

        # --- 3. Determine which bin corresponds to the true label ---

        # The true label R_r corresponds to the r-th bin (P(R_r)).
        # Map the true rating (label) to its index (r-1).
        # Assuming fixed rating values R = {R_min, R_min+R_step, ..., R_max}

        # True label index (0 to N-1) for a fixed rating set (MovieLens)
        label_index = torch.round((labels - self.rmin) / self.R_step).long().clamp(0, self.n_ratings - 1)

        # Gather the predicted probability for the true bin (P(R_r))
        # true_bin_probs shape: (B,)
        true_bin_probs = bin_probs.gather(1, label_index.unsqueeze(1)).squeeze(1)

        # --- 4. Cross-Entropy Loss (Negative Log-Likelihood) ---

        # Clamp to avoid log(0)
        true_bin_probs = torch.clamp(true_bin_probs, min=1e-10, max=1.0)
        loss = -torch.log(true_bin_probs).mean()  # Maximize log-likelihood

        loss = loss + self.regularization()

        loss.backward()
        optimizer.step()

        return loss
