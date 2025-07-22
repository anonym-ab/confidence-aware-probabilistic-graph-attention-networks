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


def sample_item_factors(model: CBPMFModel, user_idx, item_idx, ratings, mu_v, Lambda_v):
    D = model.latent_dim
    eps = 1e-4
    I = torch.eye(D, device=model.device)

    # Sample item factors
    for j in range(model.num_items):
        idx = (item_idx == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0: continue
        Ui = model.U[user_idx[idx]]
        Rij = ratings[idx].unsqueeze(1)
        prec = model.alpha * model.gamma_v[j] * model.gamma_u[user_idx[idx]].unsqueeze(1)
        Lambda_star = Lambda_v + (Ui.T * prec.squeeze(-1)) @ Ui
        cov = torch.inverse(Lambda_star)
        mean = cov @ (Lambda_v @ mu_v.unsqueeze(1) + (Ui * (prec * Rij)).sum(dim=0, keepdim=True).T)
        # Ensure covariance is symmetric and PD before sampling
        cov = 0.5 * (cov + cov.T) + eps * I
        model.V.data[j] = MultivariateNormal(mean.squeeze(-1), cov).rsample()

def sample_user_factors(model: CBPMFModel, user_idx, item_idx, ratings, mu_u, Lambda_u):
    D = model.latent_dim
    eps = 1e-4
    I = torch.eye(D, device=model.device)

    # Sample user factors
    for i in range(model.num_users):
        idx = (user_idx == i).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue

        Vj = model.V[item_idx[idx]]
        Rij = ratings[idx].unsqueeze(1)
        prec = model.alpha * model.gamma_u[i] * model.gamma_v[item_idx[idx]].unsqueeze(1)

        Lambda_star = Lambda_u + (Vj.T * prec.squeeze(-1)) @ Vj

        # Symmetrize and jitter
        Lambda_star = 0.5 * (Lambda_star + Lambda_star.T) + eps * I
        cov = torch.inverse(Lambda_star)

        term1 = Lambda_u @ mu_u.unsqueeze(1)
        term2 = (Vj * (prec * Rij)).sum(dim=0, keepdim=True).T
        mean = cov @ (term1 + term2)

        # Final safety: symmetrize and jitter covariance
        cov = 0.5 * (cov + cov.T) + eps * I
        model.U.data[i] = MultivariateNormal(mean.squeeze(-1), cov).rsample()


def train_cbpmf(model: CBPMFModel, fit_dl, val_dl, environ, device, epochs=50, patience=8):

    model.to(device)
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
            sample_user_factors(model, user_idx, item_idx, ratings_norm, mu_u, Lambda_u)
            sample_item_factors(model, user_idx, item_idx, ratings_norm, mu_v, Lambda_v)

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

                val_loss += torch.sqrt(torch.mean((mu * model.rmax + model.rmin - ratings) ** 2))

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

