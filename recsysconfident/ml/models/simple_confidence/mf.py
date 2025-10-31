import torch
import torch.nn as nn

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.ml.models.simple_confidence.simple_conf_model import SimpleConfModel

def get_mf_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MatrixFactorizationModel(
        num_users=info.n_users,
        num_items=info.n_items,
        num_factors=64,
        rmin=info.rate_range[0],
        rmax=info.rate_range[1]
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class MatrixFactorizationModel(SimpleConfModel):

    def __init__(self, num_users: int, num_items:int, num_factors:int, rmin:float, rmax:float):
        super(MatrixFactorizationModel, self).__init__()

        self.rmin = rmin
        self.rmax = rmax

        self.user_factors = nn.Embedding(num_users, num_factors)  # User Latent Factors (stack multiple in channels)
        self.item_factors = nn.Embedding(num_items, num_factors)  # Item Latent Factors
        self.user_bias = nn.Embedding(num_users, 1)  # User Bias
        self.item_bias = nn.Embedding(num_items, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        # Initialize embeddings
        nn.init.xavier_uniform(self.user_factors.weight)
        nn.init.xavier_uniform(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.criterion = nn.MSELoss()

    def forward(self, user, item):

        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()

        dot_product = (user_embedding * item_embedding).sum(dim=1)  # Element-wise product, summed over latent factors
        prediction = dot_product + user_bias + item_bias + self.global_bias

        # ---- confidence calculation----------
        # Compute the L2 norm of each row in matrix1 and matrix2
        u_norm = torch.norm(user_embedding, p=2, dim=1)
        i_norm = torch.norm(item_embedding, p=2, dim=1)

        sim = dot_product / (u_norm * i_norm)  # Compute cosine similarity

        return torch.stack([prediction * (self.rmax - self.rmin) + self.rmin, torch.abs(sim - sim.mean())], dim=1)

    def regularization(self):
        return self.l2(self.user_factors) + self.l2(self.item_factors)
