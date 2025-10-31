import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Gamma

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.fit.early_stopping import EarlyStopping


def get_cbpmf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CBPMFModel(
        num_users=info.n_users,
        num_items=info.n_items,
        latent_dim=20,
        rmin=info.rate_range[0],
        rmax=info.rate_range[1],
        device = device,
        delta_r=0.25
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class CBPMFModel(nn.Module):
    """
    CBPMFModel stores all learnable parameters and hyperparameters.
    forward() returns predicted mean and std for given user/item indices.
    """
    def __init__(self, num_users, num_items, latent_dim, device, rmax: float=0., rmin: float=0., delta_r=0.125,
                 a_u=1.0, b_u=1.0, a_v=1.0, b_v=1.0, beta0_u=1.0, nu0_u=None, beta0_v=1.0, nu0_v=None,
                 init_alpha=1.0):
        super().__init__()

        self.train_method = train_cbpmf
        self.delta_r = delta_r
        self.rmax = rmax
        self.rmin = rmin
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.device = device

        # Latent factors
        self.U = nn.Parameter(torch.randn(num_users, latent_dim, device=device))
        self.V = nn.Parameter(torch.randn(num_items, latent_dim, device=device))
        self.alpha = nn.Parameter(torch.tensor(init_alpha, device=device))

        # Variance parameters gamma_U, gamma_V
        self.gamma_u = nn.Parameter(torch.ones(num_users, device=device))
        self.gamma_v = nn.Parameter(torch.ones(num_items, device=device))
        self.a_u, self.b_u = a_u, b_u
        self.a_v, self.b_v = a_v, b_v

        # Gaussian-Wishart hyperparameters
        D = latent_dim
        # Users
        self.mu0_u = nn.Parameter(torch.zeros(D, device=device), requires_grad=False)
        self.beta0_u = beta0_u
        self.W0_u = nn.Parameter(torch.eye(D, device=device), requires_grad=False)
        self.nu0_u = nu0_u or D
        # Items
        self.mu0_v = nn.Parameter(torch.zeros(D, device=device), requires_grad=False)
        self.beta0_v = beta0_v
        self.W0_v = nn.Parameter(torch.eye(D, device=device), requires_grad=False)
        self.nu0_v = nu0_v or D

    def forward(self, user_idx, item_idx):
        # Compute prediction mean and std
        u = self.U[user_idx]            # (batch, D)
        v = self.V[item_idx]            # (batch, D)
        dot = torch.sum(u * v, dim=1)   # (batch,)
        mean = dot
        # precision per instance
        precision = self.alpha * self.gamma_u[user_idx] * self.gamma_v[item_idx]
        std = torch.sqrt(1.0 / precision)
        return mean, std

    def predict(self, user_idx, item_idx):
        mu, sigma = self.forward(user_idx, item_idx)

        dist = torch.distributions.Normal(mu, sigma)

        pred_rating = mu * (self.rmax - self.rmin) + self.rmin
        confidence = dist.cdf(mu + self.delta_r) - dist.cdf(mu - self.delta_r)

        return pred_rating, confidence

def sample_hyper_u(model: CBPMFModel):
    # Sample Gaussian-Wishart hyperparameters for U
    N, D = model.num_users, model.latent_dim
    U = model.U.data
    U_bar = U.mean(dim=0)
    S = ((U - U_bar).T @ (U - U_bar)) / N
    beta_n = model.beta0_u + N
    mu_n = (model.beta0_u * model.mu0_u + N * U_bar) / beta_n
    nu_n = model.nu0_u + N
    W0_inv = torch.inverse(model.W0_u)
    W_n_inv = W0_inv + N * S + (model.beta0_u * N) / beta_n * torch.outer(model.mu0_u - U_bar, model.mu0_u - U_bar)
    W_n_inv += 1e-6 * torch.eye(D, device=model.device)  # Prevent singularity
    W_n = torch.inverse(W_n_inv)
    # Sample Lambda_u via Wishart (Bartlett)
    A = MultivariateNormal(torch.zeros(D, device=model.device), W_n).rsample((nu_n,))
    Lambda_u = A.T @ A
    cov_mu = torch.inverse(beta_n * Lambda_u)
    mu_u = MultivariateNormal(mu_n, cov_mu).rsample()
    return mu_u, Lambda_u


def sample_hyper_v(model: CBPMFModel):
    M, D = model.num_items, model.latent_dim
    V = model.V.data
    V_bar = V.mean(dim=0)
    S = ((V - V_bar).T @ (V - V_bar)) / M
    beta_n = model.beta0_v + M
    mu_n = (model.beta0_v * model.mu0_v + M * V_bar) / beta_n
    nu_n = model.nu0_v + M
    W0_inv = torch.inverse(model.W0_v)
    W_n_inv = W0_inv + M * S + (model.beta0_v * M) / beta_n * torch.outer(model.mu0_v - V_bar, model.mu0_v - V_bar)
    W_n_inv += 1e-6 * torch.eye(D, device=model.device)  # Prevent singularity
    W_n = torch.inverse(W_n_inv)
    A = MultivariateNormal(torch.zeros(D, device=model.device), W_n).rsample((nu_n,))
    Lambda_v = A.T @ A
    cov_mu = torch.inverse(beta_n * Lambda_v)
    mu_v = MultivariateNormal(mu_n, cov_mu).rsample()
    return mu_v, Lambda_v


def sample_gamma(model: CBPMFModel, user_idx, item_idx, ratings):
    dot = (model.U[user_idx] * model.V[item_idx]).sum(dim=1)
    err2 = (ratings - dot)**2

    # Ensure indices are int64 (long) and on the correct device
    user_idx = user_idx.to(model.device).long()
    item_idx = item_idx.to(model.device).long()

    # Compute sum_u: scatter_add alpha * gamma_v[j] * err2 per user
    gamma_v_j = model.gamma_v[item_idx]
    sum_u = torch.zeros(model.num_users, device=model.device, dtype=torch.float)
    sum_u.scatter_add_(0, user_idx, (model.alpha * gamma_v_j * err2))

    # Compute count_u using bincount
    count_u = torch.bincount(user_idx, minlength=model.num_users).float()

    # Compute sum_v: scatter_add alpha * gamma_u[i] * err2 per item
    gamma_u_i = model.gamma_u[user_idx]
    sum_v = torch.zeros(model.num_items, device=model.device, dtype=torch.float)
    sum_v.scatter_add_(0, item_idx, (model.alpha * gamma_u_i * err2))

    # Compute count_v using bincount
    count_v = torch.bincount(item_idx, minlength=model.num_items).float()

    # Sample new gamma_u and gamma_v from Gamma distributions
    model.gamma_u.data = Gamma(model.a_u + 0.5 * count_u, model.b_u + 0.5 * sum_u).rsample()
    model.gamma_v.data = Gamma(model.a_v + 0.5 * count_v, model.b_v + 0.5 * sum_v).rsample()

def sample_item_factors_sparse(
    model,
    user_idx,          # LongTensor [num_obs]
    item_idx,          # LongTensor [num_obs]
    ratings,           # Tensor [num_obs]
    mu_v,              # Tensor [D]
    Lambda_v,          # Tensor [D, D]
    item_batch_size=10000,
    obs_chunk_size=1_000_000,
    eps=1e-6
):
    """
    Batched sparse-style item factor sampling (no Python loop over items).

    Args:
      model: object with attributes:
          - U: [num_users, D]
          - alpha: scalar
          - gamma_u: [num_users]
          - gamma_v: [num_items]
          - num_items: int
          - V: tensor to write results to [num_items, D]
      user_idx, item_idx, ratings: 1D observation tensors
      mu_v: prior mean for a single item [D]
      Lambda_v: prior precision for a single item [D, D]
      item_batch_size: number of items to process per batch when solving
      obs_chunk_size: number of observations per chunk when accumulating
    """

    device = model.U.device
    dtype = model.U.dtype
    D = model.U.shape[1]
    num_items = model.num_items

    prior_term = (Lambda_v @ mu_v).to(device=device, dtype=dtype)

    # Allocate accumulators
    per_item_precision = torch.zeros((num_items, D, D), device=device, dtype=dtype)
    per_item_rhs = torch.zeros((num_items, D), device=device, dtype=dtype)

    # weights: w = alpha * gamma_v[item] * gamma_u[user]
    w_all = model.alpha * model.gamma_v[item_idx].to(device=device) * model.gamma_u[user_idx].to(device=device)
    ratings = ratings.to(device=device, dtype=dtype)
    w_all = w_all.to(device=device, dtype=dtype)

    n_obs = user_idx.shape[0]
    start = 0
    while start < n_obs:
        end = min(start + obs_chunk_size, n_obs)
        ui = user_idx[start:end]
        ii = item_idx[start:end]
        rij = ratings[start:end]
        wij = w_all[start:end]

        # Gather user vectors
        U_chunk = model.U[ui]  # [m, D]

        # Weighted outer products per obs: wij * (u_i u_i^T)
        outer = (U_chunk.unsqueeze(2) * U_chunk.unsqueeze(1)) * wij.view(-1, 1, 1)

        # Accumulate precision blocks
        per_item_precision.index_add_(0, ii, outer)

        # Accumulate rhs: (wij * rij) * u_i
        rhs_chunk = (wij * rij).unsqueeze(1) * U_chunk
        per_item_rhs.index_add_(0, ii, rhs_chunk)

        start = end

    # Add prior contributions
    per_item_precision += Lambda_v.unsqueeze(0)
    per_item_rhs += prior_term.unsqueeze(0)

    # Jitter for numerical stability
    if eps > 0:
        diag_idx = torch.arange(D, device=device)
        per_item_precision[:, diag_idx, diag_idx] += eps

    # Allocate output
    V_out = torch.empty((num_items, D), device=device, dtype=dtype)

    item_start = 0
    while item_start < num_items:
        item_end = min(item_start + item_batch_size, num_items)
        idx_slice = slice(item_start, item_end)

        P_batch = per_item_precision[idx_slice]  # [B, D, D]
        b_batch = per_item_rhs[idx_slice]        # [B, D]
        B = P_batch.shape[0]

        # Cholesky decomposition of precision
        L = torch.linalg.cholesky(P_batch)       # [B, D, D]

        # Mean = P^{-1} b
        b_batch_col = b_batch.unsqueeze(-1)      # [B, D, 1]
        mean_col = torch.cholesky_solve(b_batch_col, L)  # [B, D, 1]
        mean = mean_col.squeeze(-1)              # [B, D]

        # Sampling: z ~ N(0, I), y = L^{-1} z, sample = mean + y
        z = torch.randn((B, D), device=device, dtype=dtype)
        y_col = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False, left=True)

        y = y_col.squeeze(-1)
        sample_batch = mean + y

        V_out[idx_slice] = sample_batch

        item_start = item_end

    # Write back to model.V
    model.V.data[:] = V_out

    return model.V


def sample_user_factors_sparse(
    model,
    user_idx,         # LongTensor [num_obs]
    item_idx,         # LongTensor [num_obs]
    ratings,          # Tensor [num_obs]
    mu_u,             # Tensor [D]
    Lambda_u,         # Tensor [D, D]
    user_batch_size=10000,
    obs_chunk_size=1_000_000,
    eps=1e-6
):
    """
    Build block-diagonal precision per user using scatter (index_add) and solve batched
    linear systems with Cholesky to sample user factors.

    This avoids a Python-level loop over users. Only loops:
      - Over observation chunks (to limit memory while forming per-user sums).
      - Over user batches when solving (to limit memory during Cholesky/solve).

    Args:
      model: object with attributes:
          - V: [num_items, D]
          - alpha: scalar
          - gamma_u: [num_users]
          - gamma_v: [num_items]
          - num_users: int
          - U: tensor to write results to (num_users, D)
      user_idx, item_idx, ratings: observation arrays (1D)
      mu_u: prior mean for *a single* user (D,)
      Lambda_u: prior precision for *a single* user (D,D)
      user_batch_size: how many users to solve simultaneously (tune for memory)
      obs_chunk_size: how many observations to process per chunk when building sums
    """

    device = model.V.device
    dtype = model.V.dtype
    D = model.V.shape[1]
    num_users = model.num_users

    # Precompute constant prior term
    prior_term = (Lambda_u @ mu_u).to(device=device, dtype=dtype)   # (D,)

    # Initialize accumulators:
    # per_user_precision_acc: [num_users, D, D]
    # per_user_rhs_acc:       [num_users, D]
    per_user_precision = torch.zeros((num_users, D, D), device=device, dtype=dtype)
    per_user_rhs = torch.zeros((num_users, D), device=device, dtype=dtype)

    # weights per observation
    # w = alpha * gamma_u[user_idx] * gamma_v[item_idx]
    w_all = model.alpha * model.gamma_u[user_idx].to(device=device) * model.gamma_v[item_idx].to(device=device)
    # ensure shapes/dtypes
    w_all = w_all.to(device=device, dtype=dtype)
    ratings = ratings.to(device=device, dtype=dtype)

    n_obs = user_idx.shape[0]
    start = 0
    while start < n_obs:
        end = min(start + obs_chunk_size, n_obs)

        ui = user_idx[start:end]      # LongTensor [m]
        ii = item_idx[start:end]      # LongTensor [m]
        rj = ratings[start:end]       # [m]
        wj = w_all[start:end]         # [m]

        # gather item vectors for this chunk: [m, D]
        V_chunk = model.V[ii]         # [m, D]

        # build weighted outer products: outer = wj[:,None,None] * (V_chunk[:,:,None] * V_chunk[:,None,:])
        # shape: [m, D, D]
        # compute as (V_chunk.unsqueeze(2) * V_chunk.unsqueeze(1)) * wj[:,None,None]
        outer = (V_chunk.unsqueeze(2) * V_chunk.unsqueeze(1)) * wj.view(-1, 1, 1)

        # accumulate into per_user_precision using index_add_ on dim=0
        # index_add_ will add outer[k] into per_user_precision[ ui[k], ... ]
        per_user_precision.index_add_(0, ui, outer)

        # accumulate rhs: (wj * rj)[:,None] * V_chunk  -> shape [m, D]
        rhs_chunk = (wj * rj).unsqueeze(1) * V_chunk   # [m, D]
        per_user_rhs.index_add_(0, ui, rhs_chunk)

        start = end

    # Add prior precision and prior term for every user
    # Lambda_u: [D,D] -> expand to [num_users, D, D]
    per_user_precision += Lambda_u.unsqueeze(0)    # broadcasting

    per_user_rhs += prior_term.unsqueeze(0)        # [num_users, D]

    # Add jitter for numerical stability (to diagonal of each per-user precision)
    if eps > 0:
        diag_idx = torch.arange(D, device=device)
        per_user_precision[:, diag_idx, diag_idx] += eps

    # Solve per-user systems in batches to avoid a single huge Cholesky
    # For each batch of users: compute L = cholesky(precision), mean = cholesky_solve(rhs)
    # then sample: z ~ N(0, I) -> y = solve_triangular(L, z) -> sample = mean + y

    U_out = torch.empty((num_users, D), device=device, dtype=dtype)

    user_start = 0
    while user_start < num_users:
        user_end = min(user_start + user_batch_size, num_users)
        idx_slice = slice(user_start, user_end)

        P_batch = per_user_precision[idx_slice]   # [B, D, D]
        b_batch = per_user_rhs[idx_slice]         # [B, D]

        B = P_batch.shape[0]

        # Cholesky of precision (lower triangular L such that P = L @ L.T)
        # If some matrices are not SPD due to numerical issues, this will raise; we added jitter.
        L = torch.linalg.cholesky(P_batch)        # [B, D, D] lower-tri

        # Solve for mean: mean = P^{-1} b = cholesky_solve(b, L)
        # cholesky_solve expects b with shape [..., n, k], so expand last dim
        b_batch_col = b_batch.unsqueeze(-1)       # [B, D, 1]
        mean_col = torch.cholesky_solve(b_batch_col, L)  # [B, D, 1]
        mean = mean_col.squeeze(-1)               # [B, D]

        # Sample:
        # z ~ N(0, I) shape [B, D]
        z = torch.randn((B, D), device=device, dtype=dtype)

        # Solve L y = z^T for y: use solve_triangular for batch: returns shape [B, D, 1]
        y_col = torch.linalg.solve_triangular(L, z.unsqueeze(-1), upper=False, left=True)
        # [B, D, 1]
        y = y_col.squeeze(-1)    # [B, D]

        sample_batch = mean + y   # [B, D]

        U_out[idx_slice] = sample_batch

        user_start = user_end

    # Write back to model.U (ensure same shape)
    model.U.data[:] = U_out

    return model.U


def train_cbpmf(model: CBPMFModel, fit_dl, val_dl, environ, device, epochs=50, patience=8):

    model = model.to(device)
    early_stopping = EarlyStopping(patience=patience, path=environ.model_uri)
    history = []
    for t in range(epochs):

        train_loss = 0.
        model.train()

        for data in fit_dl:
            user_idx, item_idx, ratings = data
            user_idx, item_idx, ratings = user_idx.to(model.device), item_idx.to(model.device), ratings.to(model.device)
            ratings_norm = (ratings - model.rmin) / (model.rmax - model.rmin)

            mu_u, Lambda_u = sample_hyper_u(model)
            mu_v, Lambda_v = sample_hyper_v(model)
            sample_gamma(model, user_idx, item_idx, ratings_norm)
            sample_user_factors_sparse(model, user_idx, item_idx, ratings_norm, mu_u, Lambda_u)
            sample_item_factors_sparse(model, user_idx, item_idx, ratings_norm, mu_v, Lambda_v)

            mu, sigma = model(user_idx, item_idx)

            mu_denorm = mu * (model.rmax - model.rmin) + model.rmin
            train_loss +=  torch.sqrt(torch.mean((mu_denorm - ratings)**2))

        avg_loss = train_loss / len(fit_dl)

        val_loss = 0.
        with torch.no_grad():
            model.eval()
            for data in val_dl:
                user_idx, item_idx, ratings = data
                user_idx, item_idx, ratings = user_idx.to(model.device), item_idx.to(model.device), ratings.to(model.device)
                mu, sigma = model(user_idx, item_idx)

                val_loss += torch.sqrt(torch.mean((mu * (model.rmax - model.rmin) + model.rmin - ratings) ** 2))

        avg_vloss = val_loss / len(val_dl)

        print(f"t: {t}, Fit AVG RMSE: {avg_loss}, Val AVG RMSE: {avg_vloss}")

        history.append({
            "epoch": t + 1,
            "loss_fit": float(avg_loss),
            "loss_val": float(avg_vloss),
        })

        if early_stopping.stop(val_loss, model):
            break

    model.load_state_dict(torch.load(environ.model_uri, weights_only=True))

    return history

def inference_cbpmf(model: CBPMFModel, val_dataloader, delta_r=0.125, rmin=1, rmax=5.):

    model.eval()

    conf_tensor = []
    rating_tensor = []
    pred_rating_tensor = []

    with torch.no_grad():

        for data in val_dataloader:
            user_idx, item_idx, ratings = data
            user_idx, item_idx = user_idx.to(model.device), item_idx.to(model.device)
            ratings_norm = ((ratings - rmin) / (rmax - rmin)).to(model.device)
            mu, sigma = model(user_idx, item_idx)

            dist = torch.distributions.Normal(mu, sigma)
            confidence = dist.cdf(ratings_norm + delta_r) - dist.cdf(ratings_norm - delta_r)
            pred_ratings = mu.cpu() * rmax + rmin

            conf_tensor.append(confidence.cpu())
            rating_tensor.append(ratings)
            pred_rating_tensor.append(pred_ratings)

    return torch.concat(rating_tensor), torch.concat(pred_rating_tensor), torch.concat(conf_tensor)

