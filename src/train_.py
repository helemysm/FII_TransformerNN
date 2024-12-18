
import wandb

import glob as glob
import numpy as np
import os
import pandas as pd
import pickle

import random
import yaml
from box import box_from_file
from pathlib import Path

from tqdm import tqdm

from core.encoder_model import model_clf

from NNClassifier import NNClassifier

### torch packages
import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

#learning
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from core.learning_schedule import StepLRWithLoss

from data.lc_read_database import LCReadDatabase

name_config = '/user/yhelem/storage/utils/config.yaml'
config = box_from_file(Path(name_config), file_type='yaml')

print('read config', config)
#complete dataset

d_dir = config.dataset.data_path

lc_database = LCReadDatabase(d_path=d_dir, config=config)

if config.dataset.injected:
    train_loader, val_loader, test_loader = lc_database.dataloader_split_lc_injection()

else:
    train_loader, val_loader, test_loader = lc_database.dataloader_split_lc()


api_key_model = config.training.api_key
project_name_wandb = "ffi_model_as"

checkpoint_path = config.dataset.checkpoint_path

name_experiment = config.modelparam.name_study
filename = config.modelparam.name_study


filename = checkpoint_path+filename

config.training.name_wandb=name_experiment
config.training.name_result_txt=filename

model = model_clf(config)
optimizer_config={"lr": config.hyperparameters.lr}
optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
scheduler = StepLRWithLoss(optimizer, step_size=10, gamma=0.9, patience=5)

#from losses import SupConLoss
#temp=0.05
#criterion_scl = SupConLoss(temperature=temp)

criterion=nn.BCELoss() #nn.CrossEntropyLoss()
clf = NNClassifier(
    model,
    config,
    criterion,
    optimizer,
    scheduler,
    project_name=project_name_wandb,
    api_key=api_key_model,
    name_experiment=name_experiment
)


clf.fit({"train": train_loader,
     "val": val_loader},
    epochs=500, checkpoint_path=filename) 

atts_eval, samples_before, labels, eval_roc_auc_epoch, eval_roc_auc_list, outputs_features, x_tic_id_list = clf.evaluate(test_loader, checkpoint_path=filename)
clf.save_to_file(checkpoint_path, config)

