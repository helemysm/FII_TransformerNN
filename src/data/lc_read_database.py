import glob as glob
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import random
import yaml
from box import box_from_file
from pathlib import Path

from tqdm import tqdm
from torch.utils import data

from dataloader_injection import FFIDataLoaderInjection, DataCollatorFFI, CustomDataLoaderFFIInjected, BalancedBatchSampler

from dataloader import FFIDataLoader, DataCollatorFFI, CustomDataLoaderFFI, DataCollatorInferenceFFI

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, TensorDataset, random_split, WeightedRandomSampler
from torch.utils.data.sampler import Sampler


positive_samples = 1103
negative_samples = 3062
labels = torch.hstack((torch.ones(positive_samples), torch.zeros(negative_samples)))
class_weights = 1.0 / torch.bincount(labels.long())


class_weights = torch.tensor([0.0003, 0.0005])
sampler = WeightedRandomSampler(weights=class_weights[labels.long()], num_samples=4165, replacement=True)

class LCReadDatabase:
  
    def __init__(self, d_path='data', config=None):
        
        self.d_path_train = os.path.join(d_path,'training')
        self.d_path_validation = os.path.join(d_path,'validation')
        self.d_path_testing = os.path.join(d_path,'testing')
        self.bz_train = config.timeseries.batch_size_train
        self.bz_val = config.timeseries.batch_size_val
        self.d_path = d_path
        self.injection = config.timeseries.injection
        self.steps_per_epoch = config.training.steps_per_epoch
        
    
    def dataloader_lc(self):
        
        class_weights = torch.tensor([0.0003, 0.0005])
        sampler = WeightedRandomSampler(weights=class_weights[labels.long()], num_samples=4165, replacement=True)

        train_data_loader_pos = FFIDataLoader(filepath=os.path.join(self.d_path_train, 'positive'), positive=True, type_dataset='train')
        train_data_loader_neg = FFIDataLoader(filepath=os.path.join(self.d_path_train, 'negative'), positive=False, type_dataset='train')
        
        train_ds = ConcatDataset([train_data_loader_pos, train_data_loader_neg])


        data_collator = DataCollatorFFI()
        train_loader = data.DataLoader(train_ds, batch_size=self.bz_train, collate_fn=data_collator, drop_last=True,sampler=sampler)

        
        val_data_loader_pos = FFIDataLoader(filepath=os.path.join(self.d_path_validation, 'positive'), positive=True, type_dataset='val')
        val_data_loader_neg = FFIDataLoader(filepath=os.path.join(self.d_path_validation, 'negative'), positive=False, type_dataset='val')
        val_ds = ConcatDataset([val_data_loader_pos, val_data_loader_neg])
        data_collator = DataCollatorFFI()
        val_loader = data.DataLoader(val_ds, batch_size=self.bz_val, collate_fn=data_collator, drop_last=True, shuffle=True)


        test_data_loader_pos = FFIDataLoader(filepath=os.path.join(self.d_path_testing, 'positive'), positive=True, type_dataset='test')
        test_data_loader_neg = FFIDataLoader(filepath=os.path.join(self.d_path_testing, 'negative'), positive=False, type_dataset='test')

        test_ds = ConcatDataset([test_data_loader_pos, test_data_loader_neg])
        data_collator = DataCollatorFFI()
        test_loader = data.DataLoader(test_ds, batch_size=1, collate_fn=data_collator, drop_last=True, shuffle=True)

        return train_loader, val_loader, test_loader


    def dataloader_split_lc_injection(self):
        
        test_size = 0.085 #0.055
        validation_size = 0.08 #0.055
        filepath_positive = os.path.join(self.d_path, 'positive')        
        tic_folders_positive_list = [os.path.join(filepath_positive, f) for f in os.listdir(filepath_positive) if os.path.isdir(os.path.join(filepath_positive, f))]
        tic_folders_positive = [file for path in tic_folders_positive_list for file in glob.glob(os.path.join(path, '**'))]
        train_paths_positive, temp_positive = train_test_split(tic_folders_positive, test_size=(test_size + validation_size), random_state=123)
        validation_paths_positive, test_paths_positive = train_test_split(temp_positive, test_size=(validation_size / (test_size + validation_size)), random_state=1234)# 40

        test_size = 0.025
        validation_size = 0.025
        filepath_negative = os.path.join(self.d_path, 'negative')
        tic_folders_negative_list = [os.path.join(filepath_negative, f) for f in os.listdir(filepath_negative) if os.path.isdir(os.path.join(filepath_negative, f))]
        tic_folders_negative = [file for path in tic_folders_negative_list for file in glob.glob(os.path.join(path, '**'))]
        train_paths_negative, temp_negative = train_test_split(tic_folders_negative, test_size=(test_size + validation_size), random_state= 123) 
        validation_paths_negative, test_paths_negative = train_test_split(temp_negative, test_size=(validation_size / (test_size + validation_size)), random_state=1234) #42

        test_size = 0.017
        validation_size = 0.017
        filepath_ebs = os.path.join(self.d_path, 'targetebs')
        tic_folders_ebs_list = [os.path.join(filepath_ebs, f) for f in os.listdir(filepath_ebs) if os.path.isdir(os.path.join(filepath_ebs, f))]
        tic_folders_ebs = [file for path in tic_folders_ebs_list for file in glob.glob(os.path.join(path, '**'))]     
        train_paths_ebs, temp_ebs = train_test_split(tic_folders_ebs, test_size=(test_size + validation_size), random_state=123) 
        validation_paths_ebs, test_paths_ebs = train_test_split(temp_ebs, test_size=(validation_size / (test_size + validation_size)), random_state=123) 
        
        train_obj_positive_path = [file for path in train_paths_positive for file in glob.glob(os.path.join(path, '*.pkl'))]
        
        train_obj_nontransit_path = [file for path in train_paths_negative for file in glob.glob(os.path.join(path, '*.pkl')) if ('fp.pkl' in file or 'is.pkl' in file or 'v.pkl' in file or 'ju_j.pkl' in file or 'ju_v.pkl' in file)]

        train_obj_ebs_path = [file for path in train_paths_ebs for file in glob.glob(os.path.join(path, '*.pkl'))  if ('eb.pkl' in file)]
        
        
        validation_paths_negative.extend(validation_paths_ebs)
        test_paths_negative.extend(test_paths_ebs)
        
        train_data_loader_pos = FFIDataLoaderInjection(dir_path_positive=train_obj_positive_path, dir_path_negative=train_obj_nontransit_path, positive=True, type_dataset='train')
        
        train_data_loader_neg = FFIDataLoaderInjection(dir_path_positive=train_obj_nontransit_path, dir_path_negative=train_obj_nontransit_path, positive=False, type_dataset='train')

        train_data_loader_ebs = FFIDataLoaderInjection(dir_path_positive=train_obj_ebs_path, dir_path_negative=train_obj_nontransit_path, positive=False, type_dataset='train')
        
        
        train_ds = ConcatDataset([train_data_loader_pos, train_data_loader_neg, train_data_loader_ebs])
        dataset_concat = [train_data_loader_pos, train_data_loader_neg, train_data_loader_ebs]
        
        data_collator = DataCollatorFFI()
        num_total_samples=len(train_ds)
        
        #dataloader_custom = self.custom_dataloader(dataset, self.bz_train, data_collator)
        train_loader_custom = CustomDataLoaderFFIInjected(datasets=dataset_concat, batch_size=self.bz_train, collate_fn=data_collator, shuffle=True, nro_steps = self.steps_per_epoch)

        train_loader = data.DataLoader(train_ds, batch_size=self.bz_train, collate_fn=data_collator, drop_last=True, shuffle=True)
        
        val_data_loader_pos = FFIDataLoaderInjection(dir_path_positive=validation_paths_positive, dir_path_negative=None, positive=True, type_dataset='val')
        val_data_loader_neg = FFIDataLoaderInjection(dir_path_positive=None, dir_path_negative=validation_paths_negative, positive=False, type_dataset='val')
        val_ds = ConcatDataset([val_data_loader_pos, val_data_loader_neg])
        data_collator = DataCollatorFFI()
        val_loader = data.DataLoader(val_ds, batch_size=self.bz_val, collate_fn=data_collator, drop_last=True, shuffle=True)

        test_data_loader_pos = FFIDataLoaderInjection(dir_path_positive=test_paths_positive, dir_path_negative=None, positive=True, type_dataset='test')
        test_data_loader_neg = FFIDataLoaderInjection(dir_path_positive=None, dir_path_negative=test_paths_negative, positive=False, type_dataset='test')
        test_ds = ConcatDataset([test_data_loader_pos, test_data_loader_neg])
        data_collator = DataCollatorFFI()
        test_loader = data.DataLoader(test_ds, batch_size=1, collate_fn=data_collator, drop_last=True, shuffle=True)
        
#       return train_loader, val_loader, test_loader, train_loader_custom
        return train_loader_custom, val_loader, test_loader




    def backup_dataloader_split_lc(self):
        
        test_size = 0.05
        validation_size = 0.05
        
        filepath_positive = os.path.join(self.d_path, 'positive')
        tic_folders_positive = [os.path.join(filepath_positive, f) for f in os.listdir(filepath_positive) if os.path.isdir(os.path.join(filepath_positive, f))]
        train_paths_positive, temp_positive = train_test_split(tic_folders_positive, test_size=(test_size + validation_size), random_state=42) #random_state=42
        validation_paths_positive, test_paths_positive = train_test_split(temp_positive, test_size=(validation_size / (test_size + validation_size)), random_state=42)#random_state=42

        filepath_negative = os.path.join(self.d_path, 'negative')
        tic_folders_negative = [os.path.join(filepath_negative, f) for f in os.listdir(filepath_negative) if os.path.isdir(os.path.join(filepath_negative, f))]
        train_paths_negative, temp_negative = train_test_split(tic_folders_negative, test_size=(test_size + validation_size), random_state=42) #random_state=42
        validation_paths_negative, test_paths_negative = train_test_split(temp_negative, test_size=(validation_size / (test_size + validation_size)), random_state=42)#random_state=42


        #training
        train_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=train_paths_positive, positive=True, type_dataset='train')
        train_data_loader_neg = FFIDataLoader(filepath=None, dir_paths=train_paths_negative, positive=False, type_dataset='train')
        train_ds = ConcatDataset([train_data_loader_pos, train_data_loader_neg])
        data_collator = DataCollatorFFI()
        num_total_samples=len(train_ds)
        print('num_total_samples', num_total_samples)
        
        class_weights = torch.tensor([0.0003, 0.0005])
        positive_samples = len(train_data_loader_pos)
        negative_samples = len(train_data_loader_neg)
        labels = torch.hstack((torch.ones(positive_samples), torch.zeros(negative_samples)))
        
        sampler = WeightedRandomSampler(weights=class_weights[labels.long()], num_samples=num_total_samples, replacement=True)
        
        train_loader = data.DataLoader(train_ds, batch_size=self.bz_train, collate_fn=data_collator, drop_last=True, sampler=sampler)
        
        # validation
        val_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=validation_paths_positive, positive=True, type_dataset='val')
        val_data_loader_neg = FFIDataLoader(filepath=None, dir_paths=validation_paths_negative, positive=False, type_dataset='val')
        val_ds = ConcatDataset([val_data_loader_pos, val_data_loader_neg])
        data_collator = DataCollatorFFI()
        val_loader = data.DataLoader(val_ds, batch_size=self.bz_val, collate_fn=data_collator, drop_last=True, shuffle=True)

        # testing
        print('files to testing ', len(test_paths_positive), len(test_paths_negative))
        test_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=test_paths_positive, positive=True, type_dataset='test')
        test_data_loader_neg = FFIDataLoader(filepath=None, dir_paths=test_paths_negative, positive=False, type_dataset='test')
        test_ds = ConcatDataset([test_data_loader_pos, test_data_loader_neg])
        data_collator = DataCollatorFFI()
        test_loader = data.DataLoader(test_ds, batch_size=1, collate_fn=data_collator, drop_last=True, shuffle=True)

        return train_loader, val_loader, test_loader


    def dataloader_split_lc(self):
        
        # postive transit signal
        test_size = 0.085
        validation_size = 0.08
        filepath_positive = os.path.join(self.d_path, 'positive')        
        tic_folders_positive_list = [os.path.join(filepath_positive, f) for f in os.listdir(filepath_positive) if os.path.isdir(os.path.join(filepath_positive, f))]
        tic_folders_positive = [file for path in tic_folders_positive_list for file in glob.glob(os.path.join(path, '**'))]
        train_paths_positive, temp_positive = train_test_split(tic_folders_positive, test_size=(test_size + validation_size), random_state=1234)
        validation_paths_positive, test_paths_positive = train_test_split(temp_positive, test_size=(validation_size / (test_size + validation_size)), random_state=1234)

        
        # negative signals
        test_size = 0.025
        validation_size = 0.025
        filepath_negative = os.path.join(self.d_path, 'negative')
        tic_folders_negative_list = [os.path.join(filepath_negative, f) for f in os.listdir(filepath_negative) if os.path.isdir(os.path.join(filepath_negative, f))]
        tic_folders_negative = [file for path in tic_folders_negative_list for file in glob.glob(os.path.join(path, '**'))]
        train_paths_negative, temp_negative = train_test_split(tic_folders_negative, test_size=(test_size + validation_size), random_state=1234)
        validation_paths_negative, test_paths_negative = train_test_split(temp_negative, test_size=(validation_size / (test_size + validation_size)), random_state=1234)

        
        # eclipsing binaries
        test_size = 0.017
        validation_size = 0.017
        filepath_ebs = os.path.join(self.d_path, 'targetebs')
        tic_folders_ebs_list = [os.path.join(filepath_ebs, f) for f in os.listdir(filepath_ebs) if os.path.isdir(os.path.join(filepath_ebs, f))]
        tic_folders_ebs = [file for path in tic_folders_ebs_list for file in glob.glob(os.path.join(path, '**'))]     
        train_paths_ebs, temp_ebs = train_test_split(tic_folders_ebs, test_size=(test_size + validation_size), random_state=1234)
        validation_paths_ebs, test_paths_ebs = train_test_split(temp_ebs, test_size=(validation_size / (test_size + validation_size)), random_state=1234)
        
        
        validation_paths_negative.extend(validation_paths_ebs)
        test_paths_negative.extend(test_paths_ebs)
        
        train_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=train_paths_positive, positive=True, type_dataset='train')
        
        train_data_loader_neg = FFIDataLoader(filepath=None, dir_paths=train_paths_negative, positive=False, type_dataset='train')

        train_data_loader_ebs = FFIDataLoader(filepath=None, dir_paths=train_paths_ebs, positive=False, type_dataset='train')
        
   
        train_ds = ConcatDataset([train_data_loader_pos, train_data_loader_neg, train_data_loader_ebs])
        dataset_concat = [train_data_loader_pos, train_data_loader_neg, train_data_loader_ebs]
        
        data_collator = DataCollatorFFI()
        num_total_samples=len(train_ds)
        
        train_loader_custom = CustomDataLoaderFFI(datasets=dataset_concat, batch_size=self.bz_train, collate_fn=data_collator, shuffle=True, nro_steps = self.steps_per_epoch)

        train_loader = data.DataLoader(train_ds, batch_size=self.bz_train, collate_fn=data_collator, drop_last=True, shuffle=True)
        
        val_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=validation_paths_positive, positive=True, type_dataset='val')
        val_data_loader_neg = FFIDataLoader(filepath=None, dir_paths=validation_paths_negative, positive=False, type_dataset='val')
        val_ds = ConcatDataset([val_data_loader_pos, val_data_loader_neg])
        data_collator = DataCollatorFFI()
        val_loader = data.DataLoader(val_ds, batch_size=self.bz_val, collate_fn=data_collator, drop_last=True, shuffle=True)

        
        test_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=test_paths_positive, positive=True, type_dataset='test')
        test_data_loader_neg = FFIDataLoader(filepath=None, dir_paths=test_paths_negative, positive=False, type_dataset='test')
        test_ds = ConcatDataset([test_data_loader_pos, test_data_loader_neg])
        data_collator = DataCollatorFFI()
        test_loader = data.DataLoader(test_ds, batch_size=1, collate_fn=data_collator, drop_last=True, shuffle=True)
 

        return train_loader_custom, val_loader, test_loader
 


    def dataloader_evaluation(self, ispositive = False):

        filepath_positive = self.d_path
        
        tic_folders_positive_list = [os.path.join(filepath_positive, f) for f in os.listdir(filepath_positive) if os.path.isdir(os.path.join(filepath_positive, f))]
        test_paths_positive = [file for path in tic_folders_positive_list for file in glob.glob(os.path.join(path, '**'))]
        test_data_loader_pos = FFIDataLoader(filepath=None, dir_paths=test_paths_positive, positive=ispositive, type_dataset='test')
        test_ds = test_data_loader_pos#ConcatDataset([test_data_loader_pos, test_data_loader_neg])
        data_collator = DataCollatorInferenceFFI()
        test_loader = data.DataLoader(test_ds, batch_size=1, collate_fn=data_collator, drop_last=True, shuffle=True)

        return test_loader
    
    
