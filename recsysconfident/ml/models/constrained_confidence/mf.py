import torch
import torch.nn as nn

from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.ml.losses import RMSELoss
from recsysconfident.ml.models.torchmodel import TorchModel

def get_mf_constrained_model_and_dataloader(info: DatasetInfo):

    max_error = info.rate_range[1] - info.rate_range[0]
    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = MatrixFactorizationModel(
        num_users=info.n_users,
        num_items=info.n_items,
        num_factors=512,
        max_error=max_error
    )

    return model, fit_dataloader, eval_dataloader, test_dataloader


class MatrixFactorizationModel(TorchModel):

    def __init__(self, num_users: int, num_items: int, num_factors:int, max_error: float):
        super(MatrixFactorizationModel, self).__init__()

        self.max_error = max_error
        # User and Item Embeddings
        self.user_factors = nn.Embedding(num_users, num_factors)  # User Latent Factors (stack multiple in channels)
        self.item_factors = nn.Embedding(num_items, num_factors)  # Item Latent Factors
        self.user_bias = nn.Embedding(num_users, 1)  # User Bias
        self.item_bias = nn.Embedding(num_items, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias

        self.w_mu_u = nn.Embedding(num_users, 1)
        self.w_mu_i = nn.Embedding(num_items, 1)
        self.w_conf_u = nn.Embedding(num_users, 1)
        self.w_conf_i = nn.Embedding(num_items, 1)

        nn.init.xavier_uniform(self.w_mu_u.weight)
        nn.init.xavier_uniform(self.w_mu_i.weight)
        nn.init.xavier_uniform(self.w_conf_u.weight)
        nn.init.xavier_uniform(self.w_conf_i.weight)

        # Initialize embeddings
        nn.init.xavier_uniform(self.user_factors.weight)
        nn.init.xavier_uniform(self.item_factors.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        self.criterion = RMSELoss()

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

        mu = (self.w_mu_u(user).squeeze() + self.w_mu_i(item).squeeze()) / 2.0
        w_conf_ui = (self.w_conf_u(user).squeeze() + self.w_conf_i(item).squeeze()) / 2.0
        confidence = torch.abs(sim * w_conf_ui - mu*sim.mean())  # confidence estimation

        return prediction, confidence

    def l2(self, layer):
        l2_loss = torch.norm(layer.weight, p=2) ** 2.0  # L2 norm squared for weights
        return l2_loss

    def regularization(self):
        return self.l2(self.user_factors) + self.l2(self.item_factors)

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
        conf_penalty = torch.log10(torch.abs(error_abs - self.max_error * (1.0 - confidence)) + 1e-6)
        confidence_loss = torch.maximum(torch.zeros_like(error_abs), conf_penalty).mean()

        loss = criterion + confidence_loss * 0.4 + self.regularization() * 1e-4

        return loss

    def vloss(self, data, device):

        u_inputs, i_inputs, y = data
        u_inputs, i_inputs, labels = u_inputs.to(device), i_inputs.to(device), y.to(device)
        outputs, confidence = self.forward(u_inputs, i_inputs)
        loss = self.criterion(outputs, labels)

        return loss
