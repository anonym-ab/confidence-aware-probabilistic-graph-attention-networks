import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.torchmodel import TorchModel


def get_gaussian_conf_mf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = GaussianConfMF(info.n_users,
                           info.n_items,
                           64,
                           rmax=info.rate_range[1],
                           rmin=info.rate_range[0])
    return model, fit_dataloader, eval_dataloader, test_dataloader

def periodic_bining(x):
    return x - x.int()

def negative_samples_loss(labels_norm, bce_fn, dist, delta_t):
    labels_norm_shifted = periodic_bining(labels_norm)
    p_labels_shifted = p(labels_norm_shifted, dist, delta_t)
    negative_loss = bce_fn(p_labels_shifted, torch.zeros_like(p_labels_shifted))
    return negative_loss

def p(x, dist, delta_t):
    return torch.abs(dist.cdf(x+delta_t/2) - dist.cdf(x-delta_t/2))

def bpr_loss(p_pos, p_neg):
    diff = p_pos - p_neg
    return torch.mean(-F.logsigmoid(diff + 1e-6))

class GaussianConfMF(TorchModel):
    def __init__(self, n_users, n_items, emb_dim, rmax: float, rmin: float):
        super(GaussianConfMF, self).__init__(None)

        self.rmax = rmax
        self.rmin = rmin

        self.delta_t = 0.24999
        self.emb_dim = emb_dim

        # User and Item Embeddings
        self.u_emb = nn.Embedding(n_users, emb_dim)  # User Latent Factors (stack multiple in channels)
        self.i_emb = nn.Embedding(n_items, emb_dim)  # Item Latent Factors
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        self.w_mu_u = nn.Embedding(n_users, 1)
        self.w_mu_i = nn.Embedding(n_items, 1)
        self.w_conf_u = nn.Embedding(n_users, 1)
        self.w_conf_i = nn.Embedding(n_items, 1)

        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.standard_dist = torch.distributions.Normal(0, 1)

        # Initialize wights
        nn.init.xavier_uniform(self.u_emb.weight)
        nn.init.xavier_uniform(self.i_emb.weight)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)
        nn.init.constant_(self.w_mu_u.weight, 0.5)
        nn.init.constant_(self.w_mu_i.weight, 0.5)
        nn.init.xavier_uniform(self.w_conf_u.weight)
        nn.init.xavier_uniform(self.w_conf_i.weight)

    def l2_bias(self, layer):
        l2_loss = self.l2(layer)
        l2_loss += torch.norm(layer.bias, p=2) ** 2
        return l2_loss

    def l2(self, layer):
        l2_loss = torch.norm(layer.weight, p=2) ** 2  # L2 norm squared for weights
        return l2_loss

    def forward(self, user_ids, item_ids):

        u_x = self.u_emb(user_ids)
        i_x = self.i_emb(item_ids)
        user_bias = self.u_bias(user_ids).squeeze()
        item_bias = self.i_bias(item_ids).squeeze()

        product = (u_x * i_x)
        dot_product = product.sum(dim=1)

        u_norm = torch.norm(u_x, p=2, dim=1)
        i_norm = torch.norm(i_x, p=2, dim=1)
        sim = dot_product / (u_norm * i_norm)

        sim_mu = (self.w_mu_u(user_ids).squeeze() + self.w_mu_i(item_ids).squeeze()) / 2.0
        w_conf_ui = (self.w_conf_u(user_ids).squeeze() + self.w_conf_i(item_ids).squeeze()) / 2.0

        c = torch.abs(w_conf_ui * sim - sim_mu)
        mu = dot_product + user_bias + item_bias + self.global_bias
        sigma = -torch.log(torch.clamp(c, min=0.00001, max=0.99999))
        # sigma = torch.sqrt(self.delta_t / (2 * self.standard_dist.icdf((1 + c) / 2)))
        outputs = torch.stack([mu, sigma], dim=1)
        return outputs

    def predict(self, user_ids, item_ids,):
        outputs = self.forward(user_ids, item_ids)
        mu, sigma = outputs[:, 0], outputs[:, 1]

        mu_norm = (mu - self.rmin) / (self.rmax - self.rmin)
        dist = torch.distributions.Normal(mu_norm.detach(), sigma)

        mu_norm = torch.clamp((mu - self.rmin) / (self.rmax - self.rmin), min=0, max=1)
        p_labels = p(mu_norm, dist, self.delta_t)

        return torch.clamp(mu, min=self.rmin, max=self.rmax), p_labels

    def regularization(self):

        return self.l2(self.u_emb) + self.l2_bias(self.u_emb)

    def eval_loss(self, user_ids, item_ids, labels):

        outputs = self.forward(user_ids, item_ids)
        mu = torch.clamp(outputs[:, 0], min=self.rmin, max=self.rmax)
        return torch.sqrt(torch.nn.functional.mse_loss(labels, mu, reduction='mean'))

    def _freeze_mf(self):
        self.u_emb.requires_grad_(False)
        self.i_emb.requires_grad_(False)
        self.u_bias.requires_grad_(False)
        self.i_bias.requires_grad_(False)
        self.global_bias.requires_grad_(False)

    def _unfreeze_mf(self):
        self.u_emb.requires_grad_(True)
        self.i_emb.requires_grad_(True)
        self.u_bias.requires_grad_(True)
        self.i_bias.requires_grad_(True)
        self.global_bias.requires_grad_(True)

    def _confidence_interval_loss(self, user_ids, item_ids, labels, optimizer):

        optimizer.zero_grad()
        self._freeze_mf()

        outputs = self.forward(user_ids, item_ids)  # shape: (batch, 2)
        mu, sigma = outputs[:, 0], outputs[:, 1]

        mu_norm = (mu - self.rmin) / (self.rmax - self.rmin)
        dist = torch.distributions.Normal(torch.clamp(mu_norm.detach(), min=0, max=1), sigma)

        labels_norm = (labels - self.rmin) / (self.rmax - self.rmin)
        p_labels = p(labels_norm, dist, self.delta_t)
        positive_loss = self.bce_loss(p_labels, torch.ones_like(p_labels))

        #negative_loss = negative_samples_loss(labels_norm + self.delta_t, self.bce_loss, dist, self.delta_t)
        negative_loss = bpr_loss(p_labels, p(periodic_bining(labels_norm + self.delta_t), dist, self.delta_t))
        S = 1
        for t in torch.arange(2 * self.delta_t, 1-self.delta_t, self.delta_t):
            #negative_loss = negative_loss + negative_samples_loss(labels_norm + t, self.bce_loss, dist, self.delta_t)
            negative_loss = negative_loss + bpr_loss(p_labels, p(periodic_bining(labels_norm + t), dist, self.delta_t))
            S += 1
        negative_loss = negative_loss / S

        loss = positive_loss + negative_loss
        loss.backward()
        optimizer.step()

        return loss

    def _mse_loss(self, user_ids, item_ids, labels, optimizer):
        optimizer.zero_grad()
        self._unfreeze_mf()
        outputs = self.forward(user_ids, item_ids)  # shape: (batch, 2)
        mu = outputs[:, 0]

        mse_loss = self.mse_loss(labels, mu)
        mse_loss.backward()
        optimizer.step()
        return mse_loss

    def loss(self, user_ids, item_ids, labels, optimizer):

        mse_loss = self._mse_loss(user_ids, item_ids, labels, optimizer)
        conf_loss = self._confidence_interval_loss(user_ids, item_ids, labels, optimizer)

        return mse_loss

