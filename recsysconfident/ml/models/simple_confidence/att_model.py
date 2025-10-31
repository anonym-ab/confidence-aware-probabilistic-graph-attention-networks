import torch
import torch.nn as nn
import torch.nn.functional as F

from recsysconfident.data_handling.dataloader.int_ui_ids_dataloader import ui_ids_label
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.ml.losses import RMSELoss
from recsysconfident.ml.models.torchmodel import TorchModel


def get_att_model_and_dataloader(info: DatasetInfo):

    fit_dataloader, eval_dataloader, test_dataloader = ui_ids_label(info)

    model = AttModel(info.n_users,
                         info.n_items,
                         512)
    return model, fit_dataloader, eval_dataloader, test_dataloader



class AttentionLayer(nn.Module):

    def __init__(self, emb_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.attention_dim = attention_dim

        self.key = nn.Linear(emb_dim, attention_dim)
        self.query = nn.Linear(emb_dim, attention_dim)
        self.value = nn.Linear(emb_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, emb_x, idx):

        K = self.key(emb_x)  # Compute the keys
        Q = self.query(emb_x[idx])  # Compute the queries
        V = self.value(emb_x)  # (n_entities, att_dim)

        # Compute attention scores
        attention_scores = Q @ K.T / self.emb_dim
        attention_weights = self.softmax(attention_scores) #(batch_size, att_dim)
        attended_values = attention_weights @ V

        return attended_values



class AttModel(TorchModel):
    def __init__(self, n_users, n_items, emb_dim):
        super(AttModel, self).__init__()

        self.emb_dim = emb_dim

        # User and Item Embeddings
        self.u_emb = nn.Embedding(n_users, emb_dim)  # User Latent Factors (stack multiple in channels)
        self.i_emb = nn.Embedding(n_items, emb_dim)  # Item Latent Factors
        self.u_bias = nn.Embedding(n_users, 1)  # User Bias
        self.i_bias = nn.Embedding(n_items, 1)  # Item Bias
        self.global_bias = nn.Parameter(torch.tensor(0.0))  # Global Bias
        self.att_u = AttentionLayer(emb_dim, emb_dim)
        self.att_i = AttentionLayer(emb_dim, emb_dim)
        self.w_r = nn.Linear(emb_dim, 1)

        # Initialize embeddings
        nn.init.xavier_uniform(self.u_emb.weight, std=0.01)
        nn.init.xavier_uniform(self.i_emb.weight, std=0.01)
        nn.init.xavier_uniform(self.w_r.weight)
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

    def forward(self, users, items):

        user_embedding = self.u_emb(users)
        item_embedding = self.i_emb(items)
        user_bias = self.u_bias(users)
        item_bias = self.i_bias(items)

        emb_product = user_embedding * item_embedding

        u_x = self.att_u(self.u_emb.weight, users)
        i_x = self.att_i(self.i_emb.weight, items)

        x = self.w_r(u_x * i_x)

        pred = (x.squeeze() + user_bias.squeeze() + item_bias.squeeze() + self.global_bias).squeeze()

        # confidence
        norm_product = torch.norm(user_embedding, p=2, dim=1) * torch.norm(item_embedding, p=2, dim=1)
        cosine_similarity = emb_product.sum(dim=1).squeeze() / norm_product

        return pred, cosine_similarity

    def predict(self, data, device):

        user, item, label = data
        prediction, confidence = self.forward(user.to(device), item.to(device))
        return prediction, confidence, label.to(device)

    def loss(self, data, device):

        vloss = self.vloss(data, device)
        return vloss + self.regularization() * 1e-4

    def vloss(self, data, device):

        u_inputs, i_inputs, y = data
        u_inputs, i_inputs, labels = u_inputs.to(device), i_inputs.to(device), y.to(device)
        outputs, confidence = self.forward(u_inputs, i_inputs)
        loss = self.criterion(outputs, labels)

        return loss

    def regularization(self):

        #return (self.l2_bias(self.w_u) + self.l2_bias(self.w_i) + self.l2_bias(self.w_r))
        return 0