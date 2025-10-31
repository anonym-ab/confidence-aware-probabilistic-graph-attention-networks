import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.simple_confidence.simple_conf_model import SimpleConfModel


def get_mf_cluster_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MFClustering(info.n_users,
                         info.n_items,
                         64,
                         rmin=info.rate_range[0],
                         rmax=info.ratings_df[1])
    return model, fit_dataloader, eval_dataloader, test_dataloader


class MFClustering(SimpleConfModel):
    def __init__(self, n_users:int, n_items:int, emb_dim: int, rmin:float, rmax:float):
        super(MFClustering, self).__init__()

        self.rmin = rmin
        self.rmax = rmax

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

        self.dropout1 = nn.Dropout(p=0.25)

        # Initialize embeddings
        nn.init.xavier_uniform(self.u_emb.weight)
        nn.init.xavier_uniform(self.i_emb.weight)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)

        self.criterion = nn.MSELoss()

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
        emb_weight = F.relu(W_emb(emb_weight))
        norm_embeddings = F.normalize(emb_weight, p=2, dim=1)  # Shape: (n_entities, emb_dim)
        sim_matrix = torch.matmul(norm_embeddings[idx], norm_embeddings.T)  # Shape: (batch_size, n_entities)
        batch_size = len(idx)
        mask = torch.arange(batch_size, device=emb_weight.device)  # Indices for batch dimension
        sim_matrix[mask, idx] = 0
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

        return torch.stack([pred * (self.rmax - self.rmin) + self.rmin, torch.abs(sim - sim.mean())],dim=1)

    def regularization(self):

        return 0#(self.l2_bias(self.w_u) + self.l2_bias(self.w_i)  + self.l2_bias(self.w_r))
