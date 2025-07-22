from torch import nn

class TorchModel(nn.Module):

    def __init__(self, items_per_user: dict|None):
        super(TorchModel, self).__init__()
        self.items_per_user = items_per_user

    def regularization(self):
        raise NotImplementedError("This method is not implemented yet")

    def predict(self, user_ids, item_ids):
        raise NotImplementedError("This method is not implemented yet")

    def eval_loss(self, user_ids, item_ids, labels):
        raise NotImplementedError("This method is not implemented yet")

    def loss(self, user_ids, item_ids, labels, optimizer):
        raise NotImplementedError("This method is not implemented yet")
