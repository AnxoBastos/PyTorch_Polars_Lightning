import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import lightning.pytorch as lp
import json
import numpy as np
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

class Model(lp.LightningModule):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_features=input_dim, out_features=100)
        self.layer2 = nn.Linear(in_features=100, out_features=50)
        self.layer3 = nn.Linear(in_features=50, out_features=3)
        self.metric_test = torchmetrics.regression.MeanSquaredError()
        self.loss_fn = nn.MSELoss(reduction='sum')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = model(inputs)
        loss = self.loss_fn(outputs, labels)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        vinputs, vlabels = val_batch
        voutputs = model(vinputs)
        self.metric_test(voutputs, vlabels)
        self.log("MSE/Test", self.metric_test)

    def forward(self, x): 
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) 
        return x

class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() -1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)
    
    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)
    
    def detransform(self, values):
        return values  * (self.std + self.epsilon) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    @staticmethod
    def mean_std(dataset):
        nums = dataset.select([pl.col(pl.Int8),pl.col(pl.Int16),pl.col(pl.Float32)]).drop("id").collect()
        means = torch.tensor(nums.mean().to_numpy()).squeeze()
        stds = torch.tensor(nums.std().to_numpy()).squeeze()
        return (means, stds)

    def __repr__(self):
        return f'mean: {self.mean}, std:{self.std}, epsilon:{self.epsilon}'

class SolarFlareDataset(Dataset):
    def __init__(self, src_file, transform=None, expr_dummies=None):
        self.dataset = pl.scan_csv(
            src_file, 
            has_header=False,
            skip_rows=0,
            separator=',',
            null_values='?',
            dtypes={
                "column_1": pl.Int8,
                "column_2": pl.Float32,
                "column_3": pl.Categorical,
                "column_4": pl.Categorical,
                "column_5": pl.Categorical,
                "column_6": pl.Categorical,
                "column_7": pl.Categorical,
                "column_8": pl.Categorical,
                "column_9": pl.Categorical,
                "column_10": pl.Float32,
                "column_11": pl.Float32,
                "column_12": pl.Float32,
                "column_13": pl.Float32,
                "column_14": pl.Float32,
                "column_15": pl.Categorical,
                "column_16": pl.Categorical,
                "column_17": pl.Float32,
                "column_18": pl.Categorical,
                "column_19": pl.Float32,
                "column_20": pl.Float32,
                "column_21": pl.Float32,
                "column_22": pl.Float32,
                "column_23": pl.Float32,
                "column_24": pl.Float32,
                "column_25": pl.Float32,
                "column_26": pl.Float32
            }).drop_nulls().with_row_index("id")
        
        if transform != None: 
            self.transform = transform
        else:
            mean, std = StandardScaler.mean_std(self.dataset)
            self.transform = StandardScaler(mean, std)
            with open("pytorch_polars_lightning/doc/scaler.json", "w") as json_file: 
                json.dump({'mean':mean.tolist(), 'std':std.tolist()}, json_file)

        if expr_dummies != None:
            self.expr_dummies = [(pl.col(item["name"]) == item["value"] ).alias(f'{item["name"]}-{item["value"]}') for item in expr_dummies]
        else:
            self.expr_dummies = one_hot_encoding(self.dataset)

    def __len__(self):
        return self.dataset.select(pl.len()).collect().item()
    
    def __getitem__(self, index):
        print(index)
        if torch.is_tensor(index):
            index = index.tolist()
        else:
            index = [index]
        
        data = self.dataset.filter(pl.col("id").is_in(index)).drop("id").collect()
        numeric_data = data.select([pl.col(pl.Int8),pl.col(pl.Float32)])
        numeric_tensor =  self.transform.transform(torch.tensor(numeric_data.to_numpy()).squeeze())
        categorical_data = data.select([pl.col(pl.Categorical)]).with_columns(self.expr_dummies).drop(data.select([pl.col(pl.Categorical)]).columns)
        categorical_tensor = torch.tensor(categorical_data.to_numpy().astype(np.int32)).squeeze()
        sample = torch.cat((numeric_tensor, categorical_tensor), dim=-1)
        return (sample[:-1], sample[-1:])

def one_hot_encoding(dataset):
    categorical = dataset.select([pl.col(pl.Categorical)]).drop("id").collect()
    dummies = [[{"name": columna, 'value': i} for i in categorical.get_column(columna).cat.get_categories()] for columna in categorical.columns]
    dummies_flat = [item for row in dummies for item in row]

    with open("pytorch_polars_lightning/doc/columns.json", "w") as json_file: 
        json.dump(dummies_flat, json_file)

    return [(pl.col(item["name"]) == item["value"] ).alias(f'{item["name"]}-{item["value"]}') for item in dummies_flat]
        
if __name__ == '__main__':
    
    try:
        scaler = None
        scaler_path = Path("pytorch_polars_lightning/doc/scaler.json")
        if scaler_path.is_file():
            with open(scaler_path, 'r') as json_file:
                data = json.load(json_file)
                scaler = StandardScaler(torch.tensor(data['mean']), torch.tensor(data['std']))

        expr_dummies = None
        expr_path = Path("pytorch_polars_lightning/doc/columns.json")
        if expr_path.is_file():
            with open(expr_path, 'r') as json_file:
                data = json.load(json_file)
                expr_dummies = data
    except:
        pass

    dataset = SolarFlareDataset('pytorch_polars_lightning/doc/imports-85.data', scaler, expr_dummies)

    dataset_len = len(dataset)
    train_size = int(dataset_len * 0.8)
    val_size = int(dataset_len - train_size)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_ldr = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=False)
    val_ldr = DataLoader(val_set, batch_size=4, shuffle=True, drop_last=False)

    model = Model(64)

    trainer = lp.Trainer(accelerator="cpu", num_nodes=1, devices=1, precision=32, limit_train_batches=0.5, max_epochs=100)
    trainer.fit(model, train_ldr, val_ldr)

    torch.save(model.state_dict(), "pytorch_polars_lightning/doc/automobile_weights.pt")