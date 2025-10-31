import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.losses import RMSELoss
from recsysconfident.ml.models.torchmodel import TorchModel


def get_mf_cluster_constrained_model_and_dataloader(info: DatasetInfo):

    error_max = info.rate_range[1] - info.rate_range[0]
    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MFClustering(n_users = info.n_users,
                         n_items = info.n_items,
                         emb_dim = 512,
                         max_error = error_max)
    return model, fit_dataloader, eval_dataloader, test_dataloader


class MFClustering(TorchModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, max_error: float):
        super(MFClustering, self).__init__()

        self.max_error = max_error
        self.emb_dim = emb_dim

        # User and Item Embeddings
        self.u_emb = nn.Embedding(n_users, emb_dim)  # User Latent Factors (stack multiple in channels)
        self.i_emb = nn.Embedding(n_items, emb_dim)  # Item Latent Factors
        self.u_bias = nn.Embedding(n_users, 1)  # User Bias
        self.i_bias = nn.Embedding(n_items, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias
        self.w_u = nn.Linear(emb_dim, emb_dim)
        self.w_i = nn.Linear(emb_dim, emb_dim)
        self.w_r = nn.Linear(emb_dim, 1)

        self.w_mu_u = nn.Embedding(n_users, 1)
        self.w_mu_i = nn.Embedding(n_items, 1)
        self.w_conf_u = nn.Embedding(n_users, 1)
        self.w_conf_i = nn.Embedding(n_items, 1)

        self.dropout1 = nn.Dropout(p=0.25)

        # Initialize embeddings
        nn.init.xavier_uniform(self.u_emb.weight)
        nn.init.xavier_uniform(self.i_emb.weight)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)

        self.criterion = RMSELoss()

    def l2(self, layer):
        l2_loss = torch.norm(layer.weight, p=2) ** 2  # L2 norm squared for weights
        return l2_loss

    def l2_bias(self, layer):
        l2_loss = self.l2(layer)
        l2_loss += torch.norm(layer.bias, p=2) ** 2
        return l2_loss

    def l1(self, layer):
        l_loss = torch.sum(torch.abs(layer.weight))  # L1 norm (sum of absolute values)
        return l_loss

    def l1_bias(self, layer):

        l1 = self.l1(layer)
        l1 += torch.sum(torch.abs(layer.bias))
        return l1

    def learned_cluster(self, emb_weight, W_emb, idx):
        #W_emb, shape: (emb_dim, emb_dim)
        emb_weight = W_emb(emb_weight)
        norm_embeddings = F.normalize(emb_weight, p=2, dim=1)  # Shape: (n_entities, emb_dim)
        sim_matrix = torch.matmul(norm_embeddings[idx], norm_embeddings.T)  # Shape: (batch_size, n_entities)
        similarity = F.softmax(sim_matrix, dim=1)  # Shape: (batch_size, n_entities)
        att_embeddings = torch.matmul(similarity, emb_weight)  # Shape: (batch_size, emb_dim)

        return att_embeddings

    def forward(self, users, items):

        user_embedding = self.u_emb(users)
        item_embedding = self.i_emb(items)
        user_bias = self.u_bias(users)
        item_bias = self.i_bias(items)

        emb_product = user_embedding * item_embedding

        u_x = self.learned_cluster(self.u_emb.weight, self.w_u, users)
        i_x = self.learned_cluster(self.i_emb.weight, self.w_i, items)

        x = self.w_r(u_x + i_x + emb_product)

        pred = (x.squeeze() + user_bias.squeeze() + item_bias.squeeze() + self.global_bias).squeeze()

        # confidence
        norm_product = torch.norm(user_embedding, p=2, dim=1) * torch.norm(item_embedding, p=2, dim=1)
        sim = emb_product.sum(dim=1).squeeze() / norm_product

        mu = (self.w_mu_u(users).squeeze() + self.w_mu_i(items).squeeze()) / 2.0
        w_conf_ui = (self.w_conf_u(users).squeeze() + self.w_conf_i(items).squeeze()) / 2.0
        confidence = torch.abs(sim * w_conf_ui - mu*sim.mean())  # confidence estimation

        return pred, confidence

    def predict(self, data, device):

        user, item, label = data
        prediction, confidence = self.forward(user.to(device), item.to(device))
        return prediction, confidence, label.to(device)

    def loss(self, data, device):

        u_inputs, i_inputs, y = data
        u_inputs, i_inputs, labels = u_inputs.to(device), i_inputs.to(device), y.to(device)
        outputs, confidence = self.forward(u_inputs, i_inputs)
        criterion = self.criterion(outputs, labels)

        error_abs = torch.abs(outputs - labels)
        conf_penalty = torch.log10(torch.abs(error_abs - self.max_error * (1 - confidence)) + 1e-6)
        confidence_loss = torch.maximum(torch.zeros_like(error_abs), conf_penalty).mean()

        loss = criterion + confidence_loss * 0.4 + self.regularization() * 1e-4

        return loss

    def vloss(self, data, device):

        u_inputs, i_inputs, y = data
        u_inputs, i_inputs, labels = u_inputs.to(device), i_inputs.to(device), y.to(device)
        outputs, confidence = self.forward(u_inputs, i_inputs)
        loss = self.criterion(outputs, labels)

        return loss

    def regularization(self):

        return (self.l2_bias(self.w_u) + self.l2_bias(self.w_i)
                + self.l2_bias(self.w_r))
