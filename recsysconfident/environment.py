import json
import os

import torch
from recsysconfident.ml.models.simple_model.gnn import get_gnn_model_and_dataloader

from recsysconfident.data_handling.datasets.amazon_products import AmazonProductsReader
from recsysconfident.data_handling.datasets.datasetinfo import DatasetInfo
from recsysconfident.data_handling.datasets.jester_joke_reader import JesterJokeReader
from recsysconfident.data_handling.datasets.movie_lens_reader import MovieLensReader
from recsysconfident.ml.models.distribution_based.cp_gat import get_cpgat_model_and_dataloader
from recsysconfident.ml.models.distribution_based.cbpmf import get_cbpmf_model_and_dataloader
from recsysconfident.ml.models.distribution_based.cp_mf import get_cpmf_model_and_dataloader
from recsysconfident.ml.models.distribution_based.lbd import get_lbd_model_and_dataloader
from recsysconfident.ml.models.simple_model.mf import get_mf_model_and_dataloader
from recsysconfident.ml.models.simple_model.mf_non_reg import get_mf_non_reg_model_and_dataloader
from recsysconfident.ml.models.distribution_based.ord_rec_mf import get_ordrec_model_and_dataloader


class Environment:

    def __init__(self, model_name: str,
                 database_name: str,
                 instance_dir: str,
                 batch_size: int = 1024,
                 split_position: int = -1,
                 root_path:str="./",
                 conf_calibration: bool=False,
                 min_inter_per_user: int=10):
        self.work_dir: str = None
        self.dataset_info: DatasetInfo = None
        self.batch_size = batch_size
        self.model_name = model_name
        self.database_name = database_name
        self.split_position = split_position
        self.root_path = root_path
        self.conf_calibration = conf_calibration
        self.min_inter_per_user = min_inter_per_user

        self.load_df_info()
        self.instance_dir = instance_dir
        self.model_uri = f"{self.instance_dir}/model-{self.split_position}.pth"

        self.setup_splits_path()


    def setup_splits_path(self):

        os.makedirs(name=f"{self.root_path}/runs", exist_ok=True)
        splits = os.listdir(f"{self.root_path}/runs")
        if self.split_position == -1:
            self.split_position = len(splits)

        os.makedirs(name=f"{self.root_path}/runs/data_splits/{self.database_name}/{self.split_position}", exist_ok=True)

    def load_df_info(self):

        if os.path.isfile(f"{self.root_path}/data/{self.database_name}/info.json"):

            with open(f"{self.root_path}/data/{self.database_name}/info.json") as f:
                info = json.load(f)
            self.dataset_info = DatasetInfo(**info, database_name=self.database_name, batch_size=self.batch_size, root_uri=self.root_path)
        else:
            raise FileNotFoundError("Info file does not exists. Check if the dataset name is correct.")

    def read_split_datasets(self, shuffle: bool):

        self.database_name_fn = {
            "ml-1m": MovieLensReader(self.dataset_info).read,
            "jester-joke": JesterJokeReader(self.dataset_info, "ratings.csv").read,
            "amazon-music": AmazonProductsReader(self.dataset_info).read,
            "amazon-clothing": AmazonProductsReader(self.dataset_info).read,
            "amazon-beauty": AmazonProductsReader(self.dataset_info).read,
            "amazon-clothes-shoes-jewelry": AmazonProductsReader(self.dataset_info).read
        }

        self.model_name_fn = {
            "cpgat": get_cpgat_model_and_dataloader,
            "ordrec": get_ordrec_model_and_dataloader,
            "cbpmf": get_cbpmf_model_and_dataloader,
            "lbd": get_lbd_model_and_dataloader,
            "cpmf": get_cpmf_model_and_dataloader,
            "mf": get_mf_model_and_dataloader,
            "mf-not-reg": get_mf_non_reg_model_and_dataloader,
            "gnn": get_gnn_model_and_dataloader,
        }

        if not self.database_name in self.database_name_fn:
            raise FileNotFoundError(f"Database {self.database_name} does not exist.")

        ratings_df = self.database_name_fn[self.database_name]()
        self.dataset_info.build(ratings_df, self.split_position, shuffle)
        print(f"Gathered dataset with {len(ratings_df)} interactions, {self.dataset_info.n_users} users"
              f" and {self.dataset_info.n_items} items.")

        print("Interactions dataset split.")

    def get_model_dataloaders(self, shuffle: bool) -> tuple:

        self.read_split_datasets(shuffle)
        if not self.model_name in self.model_name_fn:
            raise ValueError(f"Invalid model name: {self.model_name}")

        model, fit_dl, val_dl, test_dl = self.model_name_fn[self.model_name](self.dataset_info)

        if os.path.isfile(self.model_uri):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(self.model_uri, weights_only=True, map_location=device))
            print(f"Loaded model weights from {self.model_uri}")

        return model, fit_dl, val_dl, test_dl

