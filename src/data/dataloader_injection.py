import glob as glob
import numpy as np
import os
import pandas as pd
import pickle
import random
from tqdm import tqdm
from typing import List, Tuple
import scipy.stats
from scipy.interpolate import interp1d

### torch packages
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

#read lightcurve
import lightkurve as lk

import torch.utils.data as data
import matplotlib.pyplot as plt

class FFIDataLoaderInjection(Dataset):
    
    
    def __init__(self, filepath=None, dir_path_positive=None, dir_path_negative=None, positive=True, type_dataset='train'):

        
        self.ispositive = positive
        self.type_dataset = type_dataset
        
        if self.type_dataset == 'train':

            self.ffimage_obj_positive_path = dir_path_positive
            self.ffimage_obj_negative_path = dir_path_negative
            
            
            self.examples_positive = self.open_files(self.ffimage_obj_positive_path)
            self.examples_negative = self.open_files(self.ffimage_obj_negative_path)
            #self.examples_negative = self.open_dataset(self.ffimage_obj_negative_path)
            
            #self.combinations = [self.combine_paths(planet_path, self.ffimage_obj_negative_path) for planet_path in self.ffimage_obj_positive_path]
            self.combinations = [self.combine_paths(planet_path, self.examples_negative) for planet_path in self.examples_positive]
            
            self.ffimage_obj = [item for sublist in self.combinations for item in sublist]
            
        else:
            if self.ispositive:
                self.ffimage_obj_positive = [file for path in dir_path_positive for file in glob.glob(os.path.join(path, '*.pkl'))]
                self.ffimage_obj = self.open_files(self.ffimage_obj_positive)
            
                
            else:
                self.ffimage_obj_negative = [file for path in dir_path_negative for file in glob.glob(os.path.join(path, '*.pkl'))]
                self.ffimage_obj = self.open_files(self.ffimage_obj_negative)
            
        #print('===========================================')
        #print('an example ', self.examples_negative[0])
        #print('===========================================')
        self.ffimage_obj = np.array(self.ffimage_obj)
        
        
        
    def open_files(self, examples):
        
        data =  [np.load(x_sample,encoding='latin1', allow_pickle=True) for x_sample in examples]
        
        return data
    
    
    def combine_paths(self, positive_path: str, negative_paths: List[str]) -> List[Tuple[str, str]]:
        
        return [(positive_path, negative_path) for negative_path in negative_paths]

    def __len__(self):

        return len(self.ffimage_obj)#.shape[0]
    
    
    def __getitem__(self, idx):

        
        type_norm='normalize_11'
        
        white_noise = True
        
        if self.ispositive:
            label = torch.tensor([1]).long()#Variable(torch.tensor([1]).long())
        else:
            label = torch.tensor([0]).long()#Variable(torch.tensor([0]).long())
        
        if self.type_dataset=='train':
            
            data_positive_path, data_negative_path = self.ffimage_obj[idx]
            
            data_positive = data_positive_path
            data_negative = data_negative_path
            
            if True:# self.ispositive:
                
                flux_positive = data_positive['flux_clean']
                time_positive = data_positive['time_clean']

                flux_negative = data_negative['flux_clean']
                time_negative = data_negative['time_clean']

                flux_neg_norm = self.normalize_timeseries_all(np.array(flux_negative), type_norm)
                
                name_tic_id = str(data_positive['name_tic']) + '/'+ str(data_negative['name_tic'])#change for positive 
                
                #=====================================
                centroid = self.get_centroid(data_negative['all_centroids'])
                background = data_negative['background']
                #normalize centroids and background
                centroid_norm = self.normalize_timeseries_all(centroid, type_norm)
                
                background_norm = self.normalize_timeseries_all(background, type_norm)
                
                data_centroid= torch.from_numpy(centroid_norm.astype(np.float32))
                data_background= torch.from_numpy(background_norm.astype(np.float32))

                problematic_centroid = torch.nonzero(torch.isnan(data_centroid) | (data_centroid != data_centroid) | (data_centroid == None), as_tuple=False)
                problematic_background = torch.nonzero(torch.isnan(data_background) | (data_background != data_background) | (data_background == None), as_tuple=False)
                data_centroid[data_centroid.isnan()] = 0.0
                data_background[data_background.isnan()] = data_background.mean()
                #=====================================

                flux_pos_magnification = self.normalize_timeseries_all(np.array(flux_positive), type_norm)
                
                flux_pos_norm, time_pos_norm = self.syntetic(flux_pos_magnification, np.array(time_positive))
                new_injected_signal, new_injected_signal_time = injected_signal_funtion(flux_neg_norm, time_negative, flux_pos_norm, time_pos_norm)
                
                new_injected_signal_norm = self.normalize_timeseries_all(np.array(new_injected_signal), type_norm)
                
                # array time series to Tensor
                data_flux = torch.from_numpy(new_injected_signal_norm.astype(np.float32))
                data_time = torch.from_numpy(new_injected_signal_time.astype(np.float32))
                
            
            else:
                
                flux_negative = data_negative['flux_clean']
                time_negative = data_negative['time_clean']
                
                flux_norm = self.normalize_timeseries_all(np.array(flux_negative), type_norm)
                
                data_flux = torch.from_numpy(flux_norm.astype(np.float32))
                data_time = torch.from_numpy(time_negative.astype(np.float32))
            
            
            data_flux_augmented, data_centroid_augmented, data_background_augmented =self.choose_random_augmentation(data_flux, data_centroid, data_background, max_roll=20)
            
            if white_noise:
                data_flux_augmented = self.gaussian_noise_augmentation(data_flux_augmented)
            
            
            return data_flux_augmented.reshape(-1,1), label, data_time, name_tic_id, data_centroid_augmented.reshape(-1,1), data_background_augmented.reshape(-1,1)
            
        else: # case val and testing
            # for validation and testing
            data = self.ffimage_obj[idx]
            
            # this is for no injected light curves
            flux = data['flux_clean']
            time = data['time_clean']
            
            flux_norm = self.normalize_timeseries_all(np.array(flux), type_norm)
            
            data_flux = torch.from_numpy(flux_norm.astype(np.float32))
            data_time = torch.from_numpy(time.astype(np.float32))
            name_tic_id = data['name_tic'] 
            
            #=====================================
            #get centroids
            centroid = self.get_centroid(data['all_centroids'])
            background = data['background']
            #normalize centroids and background
            centroid_norm = self.normalize_timeseries_all(centroid, type_norm)
            
            background_norm = self.normalize_timeseries_all(background, type_norm)
            
            data_centroid= torch.from_numpy(centroid_norm.astype(np.float32))
            data_background= torch.from_numpy(background_norm.astype(np.float32))

            problematic_centroid = torch.nonzero(torch.isnan(data_centroid) | (data_centroid != data_centroid) | (data_centroid == None), as_tuple=False)
            problematic_background = torch.nonzero(torch.isnan(data_background) | (data_background != data_background) | (data_background == None), as_tuple=False)
            data_centroid[data_centroid.isnan()] = 0.0
            data_background[data_background.isnan()] = data_background.mean()
            #=====================================


            data_flux = data_flux.reshape(-1,1)
            data_centroid = data_centroid.reshape(-1,1)
            data_background = data_background.reshape(-1,1)
            
            return data_flux, label, data_time, name_tic_id, data_centroid, data_background


    

    def normalize_time_series_minmax(self,ts_tensor):
        min_val = np.min(ts_tensor)
        max_val = np.max(ts_tensor)
        normalized_series = 2 * (ts_tensor - min_val) / (max_val - min_val) - 1
        return np.array(normalized_series)

    
    def normalize_time_series(self, array):

        mean = np.mean(array)
        std = np.std(array)
        standardized_array = (array - mean) / std
    
        min_val = np.min(standardized_array)
        max_val = np.max(standardized_array)
        normalized_array = 2 * (standardized_array - min_val) / (max_val - min_val) - 1
    
        return np.array(normalized_array)
    
    def median_normalize(self,ts_tensor):
        median = np.median(ts_tensor)
        return ts_tensor - median

    def standard_normalize(self, ts_tensor):

        mean = np.mean(ts_tensor, axis=0)
        std_dev = np.std(ts_tensor, axis=0)
        normalized_data = (ts_tensor - mean) / std_dev
        return np.array(normalized_data)


    def normalize_timeseries_all(self, ts_tensor: np.ndarray, type_norm='normalize_11') -> np.ndarray:

        if type_norm=='percentiles':
            
            percentile_10 = np.percentile(ts_tensor, 10)
            percentile_90 = np.percentile(ts_tensor, 90)
            percentile_difference = percentile_90 - percentile_10
            if percentile_difference == 0:
                normalized_array = np.zeros_like(ts_tensor)
            else:
                normalized_array = ((ts_tensor - percentile_10) / (percentile_difference / 2)) - 1
            return normalized_array
        
        if type_norm=='zscore':
            mean = np.mean(ts_tensor, axis=0)
            std_dev = np.std(ts_tensor, axis=0)
            normalized_data = (ts_tensor - mean) / std_dev
            return np.array(normalized_data)
        
        if type_norm=='zscore_11':
            mean = np.mean(ts_tensor, axis=0)
            std_dev = np.std(ts_tensor, axis=0)
            normalized_data = (2*(ts_tensor - mean) / std_dev)-1
            return np.array(normalized_data)
        
        if type_norm=='normalize_11':
        
            mean = np.mean(ts_tensor)
            std = np.std(ts_tensor)
            standardized_array = (ts_tensor - mean) / std

            min_val = np.min(standardized_array)
            max_val = np.max(standardized_array)
            normalized_array = 2 * (standardized_array - min_val) / (max_val - min_val) - 1
            return np.array(normalized_array)

        
    def normalize_on_percentiles(self,array: np.ndarray) -> np.ndarray:

            percentile_10 = np.percentile(array, 10)
            percentile_90 = np.percentile(array, 90)
            percentile_difference = percentile_90 - percentile_10
            if percentile_difference == 0:
                normalized_array = np.zeros_like(array)
            else:
                normalized_array = ((array - percentile_10) / (percentile_difference / 2)) - 1
            return normalized_array

    def gaussian_noise_augmentation(self, ts_tensor: np.ndarray, noise_level=0.1):
        noise = torch.normal(0, noise_level, size=ts_tensor.size())
        augmented_ts_tensor = ts_tensor + noise

        return augmented_ts_tensor

    def random_roll_augmentation(self, ts_tensor, ts_tensor_centroid, ts_tensor_background, max_roll=10):
        roll_amount = torch.randint(low=1, high=max_roll, size=(1,)).item()
        rolled_ts = torch.roll(ts_tensor, shifts=roll_amount, dims=0)
        rolled_ts_centroid = torch.roll(ts_tensor_centroid, shifts=roll_amount, dims=0)
        rolled_ts_background = torch.roll(ts_tensor_background, shifts=roll_amount, dims=0)
        
        return rolled_ts, rolled_ts_centroid, rolled_ts_background

    def random_split_and_swap_augmentation(self, ts_tensor, ts_tensor_centroid, ts_tensor_background):
        
        split_index = torch.randint(low=1, high=len(ts_tensor), size=(1,)).item()
        
        #flux
        ts_part1 = ts_tensor[:split_index]
        ts_part2 = ts_tensor[split_index:]
        
        #centroid
        ts_part1_centroid = ts_tensor_centroid[:split_index]
        ts_part2_centroid = ts_tensor_centroid[split_index:]
        
        #background
        ts_part1_background = ts_tensor_background[:split_index]
        ts_part2_background = ts_tensor_background[split_index:]
        
        ts_combined = torch.cat([ts_part2, ts_part1])
        ts_combined_centroid = torch.cat([ts_part2_centroid, ts_part1_centroid])
        ts_combined_background = torch.cat([ts_part2_background, ts_part1_background])

        return ts_combined, ts_combined_centroid, ts_combined_background

    def mirror_augmentation(self, ts_tensor, ts_tensor_centroid, ts_tensor_background):
        
        mirrored_ts = torch.flip(ts_tensor, dims=[0])
        mirrored_ts_centroid = torch.flip(ts_tensor_centroid, dims=[0])
        mirrored_ts_bakground = torch.flip(ts_tensor_background, dims=[0])
        
        return mirrored_ts, mirrored_ts_centroid, mirrored_ts_bakground
    
    
    def noise_augmentation(self, ts_tensor, ts_tensor_centroid, ts_tensor_background):
        
        return self.gaussian_noise_augmentation(ts_tensor), ts_tensor_centroid, ts_tensor_background

    
    def choose_random_augmentation(self, ts_tensor, ts_tensor_centroid, ts_tensor_background, max_roll=10):
        
        augmentation_method = random.choice(['roll', 'split_and_swap','none']) #noise, mirror
        
        if augmentation_method == 'noise':
            return self.gaussian_noise_augmentation(ts_tensor), ts_tensor_centroid, ts_tensor_background
        
        elif augmentation_method == 'roll':
            ts_tensor_augmented, ts_tensor_centroid_augmented, ts_tensor_background_augmented =  self.random_roll_augmentation(ts_tensor, ts_tensor_centroid, ts_tensor_background, max_roll)
            return ts_tensor_augmented, ts_tensor_centroid_augmented, ts_tensor_background_augmented
        
        elif augmentation_method == 'split_and_swap':
            ts_tensor_augmented, ts_tensor_centroid_augmented, ts_tensor_background_augmented = self.random_split_and_swap_augmentation(ts_tensor, ts_tensor_centroid, ts_tensor_background)
            return ts_tensor_augmented, ts_tensor_centroid_augmented, ts_tensor_background_augmented
        
        elif augmentation_method == 'mirror':
            ts_tensor_augmented, ts_tensor_centroid_augmented, ts_tensor_background_augmented = self.mirror_augmentation(ts_tensor, ts_tensor_centroid, ts_tensor_background)
            return ts_tensor_augmented, ts_tensor_centroid_augmented, ts_tensor_background_augmented
        
        elif augmentation_method == 'none':
            return ts_tensor, ts_tensor_centroid, ts_tensor_background

        return ts_tensor, ts_tensor_centroid, ts_tensor_background


    def syntetic(self, fluxes, times):
    
        fluxes -= np.minimum(np.min(fluxes), 0) 
        flux_median = np.median(fluxes)
        normalized_fluxes = fluxes / flux_median
        relative_times = times - np.min(times)
        return normalized_fluxes, relative_times



    def get_centroid(self, all_centroids_ts):
    
        centroid = np.sqrt(all_centroids_ts.T[0]**2 + all_centroids_ts.T[1]**2 )
        return centroid
    
    

@dataclass
class DataCollatorFFI:    
    
    def __call__(self, examples):
        
        x, x_cen, x_bkg, x_time, label, att_mask, x_tic_id = self._tensorize_batch(examples)
        return x, x_cen, x_bkg, x_time, label, att_mask, x_tic_id
    
    def padding_lc(self, example, max_length):
        
        N, D = example.shape
        
        max_lenght = 1000
        
        if N < max_length :
        
            # for flux
            padding = max_lenght - len(example)
            result = torch.cat([example, example[:padding]])
            
            # for mask
            n=N#len(example)
            index_mask = torch.cat([torch.full((N,), True), torch.full((padding,), False)])
            # flux mask padding
            att_mask = result.clone().detach()
            att_mask = index_mask.to(torch.int)
            return result, att_mask
        
        else:
            result=example[:max_length]
            att_mask = result[:max_length].clone().detach() 
            att_mask[:] = 1
            att_mask = att_mask.squeeze(1)
            return result, att_mask
      
    def padding_ts_zero(self, example, max_length, type_padding = 'zero'):
        
        N, D = example.shape
        if N < max_length :
                
            padding = torch.zeros(max_length - N)
            if type_padding == 'mean':
                padding[padding==0] = example.mean()
                
            padding = padding.unsqueeze(0).transpose(0, 1)
            result = torch.cat([example, padding])
            #result = result.view(1, max_length, 1)
            
        else:
            result=example[:max_length]
        return result
    
    def padding_time(self, time_values, max_length):
        
        N = time_values.size()
        max_lenght = 1000
        
        if N[0] < max_length :

            padding_size = max_lenght - len(time_values)
            time_diff = time_values[1:] - time_values[:-1]
            additional_time_values = time_values[-1] + torch.cumsum(time_diff[-1] * torch.ones(padding_size), dim=0)
            result = torch.cat([time_values, additional_time_values])
            return result
        
        else:
            result=time_values[:max_length]
            return result
      
        
    def _tensorize_batch(self, examples):
        
        max_length_batch = 1000
        
        x = [x_sample[0][:max_length_batch] for x_sample in examples]
        
        label = [x_sample[1] for x_sample in examples]
        
        results_padding = [self.padding_lc(x_sample[0], max_length_batch) for x_sample in examples]
        
        x_time = [self.padding_time(x_sample[2], max_length_batch) for x_sample in examples]
        
        x_tic_id = [x_sample[3] for x_sample in examples]
        
        results_padding_cent = [self.padding_ts_zero(x_sample[4], max_length_batch) for x_sample in examples]
        results_padding_background = [self.padding_ts_zero(x_sample[5], max_length_batch, type_padding = 'mean') for x_sample in examples]
        
        
        x, att_mask = zip(*results_padding)
        stacked_tensor_x = torch.stack(x, dim=0)
        stacked_tensor_x_time = torch.stack(x_time, dim=0)
        
        stacked_tensor_x_cent = torch.stack(results_padding_cent, dim=0)
        stacked_tensor_x_background = torch.stack(results_padding_background, dim=0)
        
        #x = torch.stack(x, dim=0)
        att_mask =  torch.stack(att_mask, dim=0)
        
        #print('stacked_tensor_x', stacked_tensor_x.shape)
        #print('time', stacked_tensor_x_time.shape)
        
        if False:#(len(stacked_tensor_x[0]) < max_length_batch):
            
            print('YES, x < MAX LENGHT BATCH')
            stacked_tensor_x = torch.Tensor(stacked_tensor_x[0])
            stacked_tensor_x = stacked_tensor_x.transpose(0, 1)
            padding = torch.zeros(max_length_batch - stacked_tensor_x.shape[1])
            padding = padding.unsqueeze(0)
            
            stacked_tensor_x = torch.cat([stacked_tensor_x, padding], dim=1)
            stacked_tensor_x = stacked_tensor_x.view(1, max_length_batch, 1)
            
        return stacked_tensor_x, stacked_tensor_x_cent, stacked_tensor_x_background, stacked_tensor_x_time, label, att_mask, x_tic_id


class CustomDataLoaderFFIInjected(DataLoader):
    def __init__(self, datasets, batch_size, collate_fn, nro_steps, *args, **kwargs):
     
        self.batch_size = batch_size

        super(CustomDataLoaderFFIInjected, self).__init__(datasets, *args, **kwargs)
        self.true_batch_size = batch_size
        self.datasets = datasets
        self.collate_fn = collate_fn
        self.nro_steps = nro_steps

    def __iter__(self):
        
        #for _ in range(len(self)):
         for _ in range(self.nro_steps):
            
            batch = select_samples(self.datasets, self.true_batch_size)
            np.random.shuffle(batch)
            yield self.collate_fn(batch)
            
            
def select_samples(datasets, batch_size):
    selected_samples = []
    for dataset in datasets:
        num_samples = len(dataset)
        num_samples_to_select = min(num_samples, batch_size // 3)
        indices = torch.randperm(num_samples)[:num_samples_to_select]
        
        selected_samples.extend([dataset[i] for i in indices])
        
    while len(selected_samples) < batch_size:
        num_samples_to_select = batch_size - len(selected_samples)
        
        indices = torch.randperm(len(datasets[2]))[:num_samples_to_select]
        selected_samples.extend([datasets[2][i] for i in indices])
        
    return selected_samples



class BalancedBatchSampler(data.Sampler):
    def __init__(self, concat_dataset, batch_size_per_dataset):
        self.concat_dataset = concat_dataset
        self.batch_size_per_dataset = batch_size_per_dataset

        self.datasets = concat_dataset.datasets
        self.dataset_sizes = [len(dataset) for dataset in self.datasets]

        self.class_indices = []
        self.num_classes = len(self.datasets[0].targets.unique())  
        start_idx = 0
        for dataset in self.datasets:
            class_indices = [torch.where(dataset.targets == i)[0] + start_idx for i in range(self.num_classes)]
            self.class_indices.append(class_indices)
            start_idx += len(dataset)

    def __iter__(self):
        batch = []
        for _ in range(self.batch_size_per_dataset):
            dataset_idx = torch.randint(0, len(self.datasets), (1,))
            class_idx = torch.randint(0, self.num_classes, (1,))
            sample_idx = torch.randint(0, len(self.class_indices[dataset_idx][class_idx]), (1,))
            batch.append(self.class_indices[dataset_idx][class_idx][sample_idx].item())
        
        batch = [batch[i] for i in random_indices.tolist()]

        return iter(batch)

    def __len__(self):
        return min(self.dataset_sizes) // self.batch_size_per_dataset
    


def injected_signal_funtion(light_curve_fluxes, light_curve_times, signal_magnifications, signal_times, wandb_loggable_injection = None):

    minimum_light_curve_time = np.min(light_curve_times)
    relative_light_curve_times = light_curve_times - minimum_light_curve_time
    relative_signal_times = signal_times - np.min(signal_times)
    signal_time_length = np.max(relative_signal_times)
    light_curve_time_length = np.max(relative_light_curve_times)
    time_length_difference = light_curve_time_length - signal_time_length
    signal_start_offset = (np.random.random() * time_length_difference) + minimum_light_curve_time
    offset_signal_times = relative_signal_times + signal_start_offset
    
    baseline_flux = scipy.stats.median_abs_deviation(light_curve_fluxes)
    baseline_to_median_absolute_deviation_ratio = 10  # Arbitrarily chosen to give a reasonable scale.
    baseline_flux *= baseline_to_median_absolute_deviation_ratio
    #    baseline_flux = np.median(light_curve_fluxes)
    signal_fluxes = (signal_magnifications * baseline_flux) - baseline_flux
    if True:#self.out_of_bounds_injection_handling is OutOfBoundsInjectionHandlingMethod.RANDOM_INJECTION_LOCATION:
        signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=False, fill_value=0)
    elif False:#(self.out_of_bounds_injection_handling is OutOfBoundsInjectionHandlingMethod.REPEAT_SIGNAL and
          #time_length_difference > 0):
        before_signal_gap = signal_start_offset - minimum_light_curve_time
        after_signal_gap = time_length_difference - before_signal_gap
        minimum_signal_time_step = np.min(np.diff(offset_signal_times))
        before_repeats_needed = math.ceil(before_signal_gap / (signal_time_length + minimum_signal_time_step))
        after_repeats_needed = math.ceil(after_signal_gap / (signal_time_length + minimum_signal_time_step))
        repeated_signal_fluxes = np.tile(signal_fluxes, before_repeats_needed + 1 + after_repeats_needed)
        repeated_signal_times = None
        for repeat_index in range(-before_repeats_needed, after_repeats_needed + 1):
            repeat_signal_start_offset = (signal_time_length + minimum_signal_time_step) * repeat_index
            if repeated_signal_times is None:
                repeated_signal_times = offset_signal_times + repeat_signal_start_offset
            else:
                repeat_index_signal_times = offset_signal_times + repeat_signal_start_offset
                repeated_signal_times = np.concatenate([repeated_signal_times, repeat_index_signal_times])
        signal_flux_interpolator = interp1d(repeated_signal_times, repeated_signal_fluxes, bounds_error=True)
    else:
        signal_flux_interpolator = interp1d(offset_signal_times, signal_fluxes, bounds_error=True)
    interpolated_signal_fluxes = signal_flux_interpolator(light_curve_times)
    fluxes_with_injected_signal = light_curve_fluxes + interpolated_signal_fluxes
    
    return fluxes_with_injected_signal, light_curve_times

