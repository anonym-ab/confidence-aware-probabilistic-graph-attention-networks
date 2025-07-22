import torch
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.models.simple_model.simple_conf_model import SimpleConfModel


def get_gnn_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = Gnn(info.n_users,
                 info.n_items,
                 64,
                  1,
                rmin=info.rate_range[0],
                rmax=info.rate_range[1])
    return model, fit_dataloader, eval_dataloader, test_dataloader

class Gnn(SimpleConfModel):

    def __init__(self, n_users: int, n_items: int, emb_dim: int, heads: int, rmin: float, rmax: float):
        super().__init__()
        self.n_users = n_users
        self.rmin = rmin
        self.rmax = rmax

        self.ui_lookup = nn.Embedding(n_users + n_items, emb_dim)

        self.ui_gat_layer = GATConv(in_channels=emb_dim,
                                   out_channels=emb_dim,
                                   heads=heads,
                                   concat=False
                                   )

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2 * emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, 1)

        # Initialize embeddings
        nn.init.xavier_uniform(self.ui_lookup.weight)

        self.criterion = nn.MSELoss()

    def forward(self, users_ids, items_ids):

        ui_edges = torch.stack([users_ids, items_ids + self.n_users]) #(batch,),(batch,) -> (2, batch)

        ui_x = self.ui_lookup.weight
        ui_graph_emb = self.ui_gat_layer(x=ui_x, edge_index=ui_edges)  # (max_u_id+1, emb_dim)

        u_graph_emb = ui_graph_emb[ui_edges[0]]
        i_graph_emb = ui_graph_emb[ui_edges[1]]

        # ---- confidence calculation----------
        # Compute the L2 norm of each row in matrix1 and matrix2
        u_norm = torch.norm(u_graph_emb, p=2, dim=1)
        i_norm = torch.norm(i_graph_emb, p=2, dim=1)
        sim = (u_graph_emb * i_graph_emb).sum(dim=1).squeeze() / (u_norm * i_norm)  # Compute cosine similarity

        x = F.leaky_relu(self.fc1(torch.concat([u_graph_emb, i_graph_emb], dim=1)))
        x = self.dropout(x)
        x = self.fc2(x)

        prediction = x * (self.rmax - self.rmin) + self.rmin
        return torch.stack([prediction.squeeze(), torch.abs(sim - sim.mean())],dim=1)


    def regularization(self):
        return 0
