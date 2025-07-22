import torch
from recsysconfident.ml.models.torchmodel import TorchModel


def predict(model: TorchModel, data_loader, device: str = 'cpu'):
    # Create a DataLoader to iterate through the dataset
    model = model.to(device)
    model.eval()

    y_pred = torch.tensor([]).to(device)
    y_true = torch.tensor([]).to(device)
    pred_confs = torch.tensor([]).to(device)

    with torch.no_grad():
        for data in data_loader:

            users_ids, items_ids, ratings = data
            users_ids, items_ids, ratings = users_ids.to(device), items_ids.to(device), ratings.to(device)
            output, mconfs = model.predict(users_ids, items_ids)
            pred_confs = torch.cat((pred_confs, mconfs), dim=0)

            y_pred = torch.cat((y_pred, output), dim=0)
            y_true = torch.cat((y_true, ratings), dim=0)

    pred_confs = pred_confs.cpu().view(-1).numpy()
    return y_true.cpu().view(-1).numpy(), y_pred.cpu().view(-1).numpy(), pred_confs
