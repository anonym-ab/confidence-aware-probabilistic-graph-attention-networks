import torch
import torch.nn as nn
import torch.distributions as d

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_cpmf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = CPMF(
        num_users=info.n_users,
        num_items=info.n_items,
        latent_dim=20,
        rate_range=info.rate_range
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class CPMF(TorchModel):

    def __init__(self, num_users, num_items, latent_dim, rate_range: list):
        super().__init__(None)

        self.rmin = rate_range[0]
        self.rmax = rate_range[1]
        self.delta_r = rate_range[2] / 2

        # Latent factors
        self.user_factors = nn.Embedding(num_users, latent_dim)
        self.item_factors = nn.Embedding(num_items, latent_dim)

        # Variance parameters (γ_u, γ_v), initialized to 1.0
        self.user_gamma = nn.Embedding(num_users, 1)
        self.item_gamma = nn.Embedding(num_items, 1)
        nn.init.ones_(self.user_gamma.weight)
        nn.init.ones_(self.item_gamma.weight)

        self.alpha = nn.Parameter(torch.tensor(1.))

        self.lambda_u = 0
        self.lambda_v = 0

        self.switch_to_rating()

    def forward(self, user_ids, item_ids):
        u = self.user_factors(user_ids)  # (batch, k)
        v = self.item_factors(item_ids)  # (batch, k)
        dot = torch.sum(u * v, dim=1, keepdim=True)

        #bias = self.user_bias(user_ids) + self.item_bias(item_ids) + self.global_bias
        mean = dot.squeeze()  # predicted rating mean (batch, 1)

        # Softplus ensures γ > 0
        gamma_u = torch.clamp(self.user_gamma(user_ids), min=0.00001) #the article does not mention, but it does not work without.
        gamma_v = torch.clamp(self.item_gamma(item_ids), min=0.00001)
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

        loss = nll + self.regularization()
        loss.backward()
        optimizer.step()
        return loss

    def eval_loss(self, user_ids, item_ids, labels):
        pred_scores = self.forward(user_ids, item_ids)
        mu = pred_scores[:, 0]
        rating = mu * (self.rmax - self.rmin) + self.rmin
        return torch.sqrt(torch.nn.functional.mse_loss(labels, rating, reduction='mean'))

    def regularization(self, user_ids=None, item_ids=None):
        reg = 0.0

        # L2 regularization on all user factors (as defined by Gaussian priors in PMF)
        reg += self.lambda_u * torch.sum(self.user_factors.weight ** 2)

        # L2 regularization on all item factors (as defined by Gaussian priors in PMF)
        reg += self.lambda_v * torch.sum(self.item_factors.weight ** 2)

        return reg

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
