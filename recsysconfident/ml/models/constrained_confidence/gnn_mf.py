import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.losses import RMSELoss
from recsysconfident.ml.models.torchmodel import TorchModel


def get_gnn_mf_constrained_model_and_dataloader(info: DatasetInfo):

    max_error = info.rate_range[1] - info.rate_range[0]
    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = GnnMF(info.n_users,
                 info.n_items,
                 512,
                  1,
                  max_error)
    return model, fit_dataloader, eval_dataloader, test_dataloader

class GnnMF(TorchModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, heads: int, max_error: float):
        super().__init__()
        self.n_users = n_users
        self.max_error = max_error
        self.ui_lookup = nn.Embedding(n_users + n_items, emb_dim)

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                   out_channels=emb_dim,
                                   heads=heads,
                                   concat=False
                                   )

        self.user_bias = nn.Embedding(n_users, 1)  # User Bias
        self.item_bias = nn.Embedding(n_items, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        self.w_mu_u = nn.Embedding(n_users, 1)
        self.w_mu_i = nn.Embedding(n_items, 1)
        self.w_conf_u = nn.Embedding(n_users, 1)
        self.w_conf_i = nn.Embedding(n_items, 1)

        # Initialize embeddings
        nn.init.xavier_uniform(self.ui_lookup.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        nn.init.xavier_uniform(self.ui_lookup.weight)
        nn.init.xavier_uniform(self.w_mu_u.weight)
        nn.init.xavier_uniform(self.w_mu_i.weight)
        nn.init.xavier_uniform(self.w_conf_u.weight)
        nn.init.xavier_uniform(self.w_conf_i.weight)

        self.criterion = RMSELoss()

    def forward(self, users_ids, items_ids):

        ui_edges = torch.stack([users_ids, items_ids + self.n_users]) #(batch,),(batch,) -> (2, batch)

        ui_x = self.ui_lookup.weight
        ui_graph_emb = self.ui_gat_layer(x=ui_x, edge_index=ui_edges)  # (max_u_id+1, emb_dim)

        u_graph_emb = F.leaky_relu(ui_graph_emb[ui_edges[0]])
        i_graph_emb = F.leaky_relu(ui_graph_emb[ui_edges[1]])

        dot_product = (u_graph_emb * i_graph_emb).sum(dim=1).squeeze()

        user_bias = self.user_bias(users_ids).squeeze()
        item_bias = self.item_bias(items_ids).squeeze()
        prediction = dot_product + user_bias + item_bias + self.global_bias

        # ---- confidence calculation----------
        # Compute the L2 norm of each row in matrix1 and matrix2
        u_norm = torch.norm(u_graph_emb, p=2, dim=1)
        i_norm = torch.norm(i_graph_emb, p=2, dim=1)

        sim = dot_product / (u_norm * i_norm)  # Compute cosine similarity
        mu = (self.w_mu_u(users_ids).squeeze() + self.w_mu_i(items_ids).squeeze()) / 2.0
        w_conf_ui = (self.w_conf_u(users_ids).squeeze() + self.w_conf_i(items_ids).squeeze()) / 2.0
        confidence = torch.abs(sim * w_conf_ui - mu*sim.mean())  # confidence estimation

        return prediction, confidence

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
        return 0
