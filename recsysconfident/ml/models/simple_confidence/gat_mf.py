import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.simple_confidence.simple_conf_model import SimpleConfModel


def get_gat_mf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = GATMF(info.n_users,
                  info.n_items,
                  64,
                  1,
                  rate_range=info.rate_range)
    return model, fit_dataloader, eval_dataloader, test_dataloader

class GATMF(SimpleConfModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, heads: int, rate_range:list):
        super().__init__()
        self.n_users = n_users
        self.rmin = rate_range[0]
        self.rmax = rate_range[1]

        self.ui_lookup = nn.Embedding(n_users + n_items, emb_dim)

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                   out_channels=emb_dim,
                                   heads=heads,
                                   concat=False
                                   )

        self.user_bias = nn.Embedding(n_users, 1)  # User Bias
        self.item_bias = nn.Embedding(n_items, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        # Initialize embeddings
        nn.init.xavier_uniform(self.ui_lookup.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.criterion = nn.MSELoss()

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

        return torch.stack([prediction * (self.rmax - self.rmin) - self.rmin, torch.abs(sim - sim.mean())], dim=1)

    def regularization(self):
        return 0
