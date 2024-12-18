import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import math
import os
import time
import tqdm
import pandas as pd
from typing import Dict
from copy import deepcopy
from typing import Dict

import pickle


import glob as glob

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from utils.WandbLogger import WandbLogger
import plotly.graph_objects as go

import logging

from core.losses import SupConLoss, SupervisedContrastiveLossSIM

#from pytorch_metric_learning.losses import SupConLoss


gray_color_with_opacity = 'rgba(169, 169, 169, 0.7)' 

class NNClassifier:

    
    def __init__(self, model, config, criterion, optimizer, scheduler, project_name=None, api_key=None, name_experiment=None) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.wandb_logger = WandbLogger(project_name, api_key)
        self.wandb_logger.log_parameters(config)
        self.wandb_logger.set_name(name_experiment)
        self._start_epoch = 0
        self.hyper_params = {"epochs": self._start_epoch}
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        #self.criterion_scl = SupConLoss(temperature=0.1)
        self.criterion_scl = SupConLoss(temperature=0.1)
        
        self.scheduler = scheduler
        
        self.include_mask = config.modelparam.mask
        self._start_epoch = 0
        self.steps_per_epoch = config.training.steps_per_epoch
        self._is_parallel = False

        self.wandb_logger.log_model(self.model)
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._is_parallel = True

            notice = "Running on {} GPUs.".format(torch.cuda.device_count())
            print("\033[33m" + notice + "\033[0m")

    def fit(self, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True) -> None:

        
        len_of_train_dataset = len(loader["train"].dataset)
        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset
        
        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

            
        epochs = epochs + self._start_epoch
        

        total_steps = min(self.steps_per_epoch, len(loader["train"]))

        print_freq = 10
        steps_iter = 0
        steps_iter_val = 0
        acc_best = 0
        
        train_loss_list = []
        train_accuracy_list = []
        train_roc_auc_list = []

        val_loss_list = []
        val_accuracy_list = []
        val_roc_auc_list = []
        
        
        lambda_ = self.config.training.lambda_value
        
        ##lambda_ = 0.1
        
        val_roc_auc_epoch_list = []
        
        for epoch in range(self._start_epoch, epochs):
            
            print('epoch ', (epoch % 100 == 0))
            train_loss = 0.0

            correct = 0.0
            total = 0.0
            train_rocauc = 0.0
            self.model.train()
            
            #pbar = tqdm(total=len_of_train_dataset)
            pbar = tqdm(total=(total_steps*self.config.timeseries.batch_size_train))

            #for x, x_time, y, mask in loader["train"]: 
            for batch_idx, (x, x_cen, x_bkg, x_time, y, mask, x_tic_id) in enumerate(loader["train"]):
     
                b_size = len(y)#y.shape[0]
                total += len(y)#y.shape[0]

                x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                x_cen = x_cen.to(self.device) if isinstance(x_cen, torch.Tensor) else [i.to(self.device) for i in x_cen]
                x_bkg = x_bkg.to(self.device) if isinstance(x_bkg, torch.Tensor) else [i.to(self.device) for i in x_bkg]
                #x_time = x_time.to(self.device) if isinstance(x_time, torch.Tensor) else [i.to(self.device) for i in x_time]

                y = torch.Tensor(y).long() 
                y = y.to(self.device)
                y = y.reshape(-1)

                pbar.set_description(
                    "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                )
                pbar.update(b_size)

                self.optimizer.zero_grad()

                if self.include_mask:
                    outputs, atts = self.model(x, mask)
                else:
                    outputs, atts, output_features = self.model(x, x_cen, x_bkg, x_time)
                
                outputs = outputs.squeeze()
                
                if self.model.config.training.cl_loss:
                    loss_ce = self.criterion(outputs, y.float())
                    loss_scl = self.criterion_scl(output_features, y)
                    #print('loss_scl', loss_scl)
                    #L = (1 − λ)LCE + λLSCL
                    loss = ((1 - lambda_)*loss_ce) + (lambda_*loss_scl)
                    
                else:
                    loss = self.criterion(outputs, y.float())

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                train_predicted = (outputs > 0.5).float()
                correct += (train_predicted == y.float()).sum().cpu().item()

                train_roc_auc = roc_auc_score(y.cpu().numpy(), outputs.cpu().detach().numpy())

                
                if batch_idx % print_freq == print_freq - 1:
                    metrics_value = {"train_loss_step": float(loss.cpu().item()), "train_accuracy_step": float(correct / total), "training_roc_auc_step": float(train_roc_auc)}
                    self.wandb_logger.log_metrics_dict(metrics_value)
                
                if False:#batch_idx % print_freq == print_freq - 1:
                        self.wandb_logger.log_lightcurve_scatter(x, x_time, train_predicted, y.float(), epoch=epoch + 1, step=steps_iter)

                print('===== train =====')
                print('step ', steps_iter)

                print( "Train: " "Loss: {:.7f}, "    "Accuracy: {:.2%}, " "ROC AUC: {:.4f}".format(loss.cpu().item(), float(correct / total), train_roc_auc))

                steps_iter = steps_iter + 1

                train_loss_list.append(loss.cpu().item())
                train_accuracy_list.append(float(correct / total))
                train_roc_auc_list.append(train_roc_auc)
                
            acc_train = float(correct / total)
            avg_train_loss = train_loss / len(loader["train"])
            metrics_value = {"avg_train_loss_epoch": float(avg_train_loss), "avg_train_accuracy_epoch": float(acc_train), "avg_train_roc_auc_epoch": float(train_roc_auc)}
            
            self.wandb_logger.log_metrics_dict(metrics_value)

            self.scheduler.step(avg_train_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Current learning rate: {current_lr}")

            print(f'Epoch {epoch + 1}, Loss: {avg_train_loss:.4f}')
            
            
            print('========================== validation =========================== ')
            
            y_true_val = []  
            y_scores_val = []  
            y_pred_val = [] 
            if validation:
            #
                with torch.no_grad():
                    val_correct = 0.0
                    val_total = 0.0
                    total_val_loss = 0.0

                    val_roc_auc_list = []

                    self.model.eval()
                    for x_val, x_cen_val, x_bkg_val, x_time_val, y_val, mask_val, x_tic_id_val in loader["val"]: # 

                        val_total += len(y_val)#.shape[0]
                        x_val = x_val.to(self.device) if isinstance(x_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_val]
                        x_cen_val = x_cen_val.to(self.device) if isinstance(x_cen_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_cen_val]
                        x_bkg_val = x_bkg_val.to(self.device) if isinstance(x_bkg_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_bkg_val]
                        #x_time_val = x_time_val.to(self.device) if isinstance(x_time_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_time_val]

                        y_val = torch.Tensor(y_val).long()
                        y_val = y_val.to(self.device)
                        #y_val = y_val.reshape(-1)

                        #========================================================================
                        if self.include_mask:
                            val_output, atts_val = self.model(x_val, mask_val)
                        else:
                            val_output, atts_val, output_features_val = self.model(x_val, x_cen_val, x_bkg_val, x_time_val)#, mask_val)
                        val_output = val_output.reshape(-1)

                        if self.model.config.training.cl_loss:
                            
                            val_loss_ce = self.criterion(val_output, y_val.float())
                            val_loss_scl = self.criterion_scl(output_features_val, y_val)
                            val_loss = ((1 - lambda_)*val_loss_ce) + (lambda_*val_loss_scl)
                            
                        else:
                            val_loss = self.criterion(val_output, y_val.float())


                        total_val_loss += val_loss.cpu().item()

                        val_pred = (val_output > 0.5).float()
                        val_correct += (val_pred == y_val.float()).sum().cpu().item()

                        try:
                            val_roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.cpu().numpy())

                        except ValueError:
                            print(" Only one class present in y_true.")
                            continue

                        val_roc_auc = roc_auc_score(y_val.cpu().numpy(), val_output.cpu().numpy())
                        
                        print_freq = 10
                        if batch_idx % print_freq == print_freq - 1:

                            metrics_value = {"val_loss_step": float(val_loss.cpu().item()), "val_accuracy_step": float(val_correct / val_total), "val_roc_auc_step": float(val_roc_auc)}
                            self.wandb_logger.log_metrics_dict(metrics_value)

                        print('==== validation ==== steps : ', steps_iter_val)
                        print( "Validation: " "Loss: {:.4f}, " "Accuracy: {:.2%}, " "ROC AUC: {:.4f}".format(val_loss.cpu().item(), float(val_correct/val_total), float(val_roc_auc)) )

                        steps_iter_val = steps_iter_val + 1

                        y_true_val.extend(y_val.cpu().numpy())
                        y_scores_val.extend(val_output.cpu().numpy())
                        y_pred_val.extend(val_pred.cpu().numpy())


                    acc = float(val_correct / val_total)#self.experiment.get_metric("accuracy")

                    epoch_roc_auc = roc_auc_score(y_true_val, y_scores_val)


                    avg_val_loss = total_val_loss / len(loader["val"])
                    avg_val_accuracy = 100 * (val_correct / val_total)

                    metrics_value = {"avg_val_loss_epoch": float(avg_val_loss), "avg_val_accuracy_epoch": float(acc), "avg_val_roc_auc_epoch": float(epoch_roc_auc)}
            
                    self.wandb_logger.log_metrics_dict(metrics_value)
                    

                    print('==== validation =====')
                    batch_size_val = len(loader["val"])
                    print('batch_size_val : ', batch_size_val)
                    print(f"Epoch-first {epoch + 1}/{epochs} - Val loss: {total_val_loss / batch_size_val} - Val accuracy: {val_correct / val_total * 100:.2f}% - ROC AUC: {epoch_roc_auc:.4f}")


                    filename = checkpoint_path

                    conf_matrix_norm = confusion_matrix(np.array(y_true_val), np.array(y_pred_val), normalize='true')
                    
                    f1_score_val = f1_score(np.array(y_true_val), np.array(y_pred_val))
                    
                    
                    conf_matrix = confusion_matrix(np.array(y_true_val), np.array(y_pred_val))
                    print('===============================cm=================================')
                    print(conf_matrix_norm)
                    print(conf_matrix)
                    print('===============================cm=================================')


            pbar.close()
            
            current_memory_allocated = torch.cuda.memory_allocated()

            peak_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Current CUDA memory allocated: {current_memory_allocated / 1024**2:.2f} MB")
            print(f"Peak CUDA memory allocated: {peak_memory_allocated / 1024**2:.2f} MB")


        return train_loss_list,  train_accuracy_list, train_roc_auc_list# ''#all_attn_list, atts_val

        
    def evaluate(self, loader: Dict[str, DataLoader], verbose: bool = False, checkpoint_path: str = None) -> None or float:
 
        running_loss = 0.0
        running_corrects = 0.0
        pbar = tqdm(total=len(loader.dataset))

        atts_eval_list = []
        self.model.eval()
        
        samples = []
        samples_before = []
        labels = []
        steps_iter_test = 0
        outputs_features_list = []
        x_tic_id_list = []
        #lambda_ = 0.1
        lambda_ = self.config.training.lambda_value
        
        #with self.experiment.test():
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            eval_roc_auc_list = []     
            for x, x_cen, x_bkg, x_time, y, mask, x_tic_id in (loader):

                b_size = len(y)#y.shape[0]
                total += len(y)#y.shape[0]
                samples_before.append(x)

                x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                x_cen = x_cen.to(self.device) if isinstance(x_cen, torch.Tensor) else [i.to(self.device) for i in x_cen]
                x_bkg = x_bkg.to(self.device) if isinstance(x_bkg, torch.Tensor) else [i.to(self.device) for i in x_bkg]
                #x_time = x_time.to(self.device) if isinstance(x_time, torch.Tensor) else [i.to(self.device) for i in x_time]

                y = torch.Tensor(y).long() 
                y = y.to(self.device)
                #y = y.reshape(-1) # 

                pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                pbar.update(b_size)
                
                if self.include_mask:
                    outputs, atts_eval, outputs_features = self.model(x, mask)
                else:
                    outputs, atts_eval, outputs_features = self.model(x, x_cen, x_bkg, x_time)

                outputs = outputs.reshape(-1)
                
                if self.model.config.training.cl_loss:

                    loss_ce = self.criterion(outputs, y.float())
                    loss_scl = self.criterion_scl(outputs_features, y)
                    loss = ((1 - lambda_)*loss_ce) + (lambda_*loss_scl)
                    
                else:
                    loss = self.criterion(outputs, y.float())
                

                predicted = (outputs > 0.5).float()
                correct += (predicted == y.float()).sum().cpu().item()
                #=====================================================

                atts_eval_list.append(atts_eval)
                samples.append(x)
                labels.append([y, predicted])
                outputs_features_list.append(outputs_features)
                x_tic_id_list.append(x_tic_id)
                eval_roc_auc_list.append([y.cpu().item(), outputs.cpu().item(), predicted.cpu().item()])

                running_loss += loss.cpu().item()
                running_corrects += torch.sum(predicted == y).float().cpu().item()
                
                print(  "Loss batch: {:.6f}, "  "Loss Total: {:.6f}, " "Running Accuracy: {:.2%}, ".format(loss.cpu().item(), float(running_loss / total), float(running_corrects / total)) )


                steps_iter_test = steps_iter_test + 1

            eval_roc_auc_epoch = roc_auc_score(np.array(eval_roc_auc_list)[:,0], np.array(eval_roc_auc_list)[:,1])

            metrics_value = {"eval_accuracy": float(running_corrects/total), "eval_roc_auc": float(eval_roc_auc_epoch), "eval_avg_loss": float(running_loss/total)}
            
            self.wandb_logger.log_metrics_dict(metrics_value)
                                
            filename = checkpoint_path

            conf_matrix_norm = confusion_matrix(np.array(eval_roc_auc_list)[:,0], np.array(eval_roc_auc_list)[:,2], normalize='true')
            conf_matrix = confusion_matrix(np.array(eval_roc_auc_list)[:,0], np.array(eval_roc_auc_list)[:,2])
            
            f1_score_eval = f1_score(np.array(eval_roc_auc_list)[:,0], np.array(eval_roc_auc_list)[:,2])
                    
                
            print('===============================cm eval=================================')
            print(conf_matrix_norm)
            print(conf_matrix)
            print('===============================cm eval=================================')

            pbar.close()
        
            print("\033[33m" + "Evaluation finished. " + "\033[0m" + "Accuracy: {:.4f}".format(float(running_corrects/total)), "Loss {:.4f}".format(float(running_loss/total)), "Auc roc: {:.4f}".format(eval_roc_auc_epoch))

        
        self.wandb_logger.finish_wb()
        
        return atts_eval_list, samples_before, labels, eval_roc_auc_epoch, eval_roc_auc_list, outputs_features_list, x_tic_id_list

    def inference_single_lc(self, x, x_cen, x_bkg, x_time):
 
        with torch.no_grad():
        
            x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
            x_cen = x_cen.to(self.device) if isinstance(x_cen, torch.Tensor) else [i.to(self.device) for i in x_cen]
            x_bkg = x_bkg.to(self.device) if isinstance(x_bkg, torch.Tensor) else [i.to(self.device) for i in x_bkg]

            outputs, atts_eval, outputs_features = self.model(x, x_cen, x_bkg, x_time)

            outputs = outputs.reshape(-1)
    
        return outputs


    
    def inference(self, loader: Dict[str, DataLoader], verbose: bool = False, checkpoint_path: str = None, sector=None, sector_section = None) -> None or float:
 
        running_loss = 0.0
        running_corrects = 0.0
        pbar = tqdm(total=len(loader.dataset))
        steps_iter_inference = 0

        atts_eval_list = []
        self.model.eval()
        
        samples = []
        samples_tic_ids = []
        samples_before = []
        labels = []
        steps_iter_test = 0
        outputs_features_list = []
        lambda_ = 0.1
        #lambda_ = self.config.training.lambda

        path_dest = '/user/yhelem/storage/ffi_nn/predictions/'

        with torch.no_grad():
            correct = 0.0
            total = 0.0
            eval_roc_auc_list = []     
            for x, x_cen, x_bkg, x_time, y, mask, x_tic_id in (loader):

                b_size = len(y)#y.shape[0]
                total += len(y)#y.shape[0]
                samples_before.append(x)
                
                samples_tic_ids.append(x_tic_id)
                x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to(self.device) for i in x]
                x_cen = x_cen.to(self.device) if isinstance(x_cen, torch.Tensor) else [i.to(self.device) for i in x_cen]
                x_bkg = x_bkg.to(self.device) if isinstance(x_bkg, torch.Tensor) else [i.to(self.device) for i in x_bkg]

                y = torch.Tensor(y).long() 
                y = y.to(self.device)
                #y = y.reshape(-1)

                pbar.set_description("\033[32m"+"Evaluating"+"\033[0m")
                pbar.update(b_size)
                
                if self.include_mask:
                    outputs, atts_eval, outputs_features = self.model(x, mask)
                else:
                    outputs, atts_eval, outputs_features = self.model(x, x_cen, x_bkg, x_time)

                outputs = outputs.reshape(-1)
                
                #if self.model.config.training.cl_loss:

                #    loss_ce = self.criterion(outputs, y.float())
                #    loss_scl = self.criterion_scl(outputs_features, y)
                #    loss = ((1 - lambda_)*loss_ce) + (lambda_*loss_scl)
                #else:
                #   loss = self.criterion(outputs, y.float())
                

                predicted = (outputs > 0.5).float() # subir la prob
                correct += (predicted == y.float()).sum().cpu().item()
                #=====================================================

                labels.append([y, predicted])
                print('predicted - real', predicted.cpu().item(), y.cpu().item())
                eval_roc_auc_list.append([y.cpu().item(), outputs.cpu().item(), predicted.cpu().item()])

                running_corrects += torch.sum(predicted == y).float().cpu().item()
                
                print(  "Running Accuracy: {:.2%}, ".format(float(running_corrects / total)) )


                steps_iter_test = steps_iter_test + 1
               

            metrics_value = {"eval_accuracy": float(running_corrects/total)}
            

            filename = checkpoint_path#+filename

            conf_matrix_norm = confusion_matrix(np.array(eval_roc_auc_list)[:,0], np.array(eval_roc_auc_list)[:,2], normalize='true')
            conf_matrix = confusion_matrix(np.array(eval_roc_auc_list)[:,0], np.array(eval_roc_auc_list)[:,2])
            print('===============================cm eval=================================')
            print(conf_matrix_norm)
            print(conf_matrix)
            print('===============================cm eval=================================')

            with open(filename, "a") as file:
                file.write("--------------------EVALUATION INFERENCE-------------------\n")
                file.write(f"Accuracy: {float(running_corrects/total)}\n")
                file.write(f"Confusion Matrix norm:\n{conf_matrix_norm}\n")
                file.write(f"Confusion Matrix:\n{conf_matrix}\n")
                file.write("---------------------------------------\n")

            pbar.close()
        #acc = self.wandb_logger.get_metric("accuracy")

            steps_iter_inference = steps_iter_inference+1
            
            print("\033[33m" + "Inference finished. " + "\033[0m" + "Accuracy: {:.4f}".format(float(running_corrects/total)), "Loss {:.4f}".format(float(running_loss/total)))

            
        return atts_eval_list, samples_before, labels, [], eval_roc_auc_list, outputs_features_list, samples_tic_ids

    
    def save_checkpoint(self) -> dict:

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict())
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model.state_dict())

        return checkpoints

    def save_to_file(self, path: str, config) -> str:
        
        if not os.path.isdir(path):
            os.mkdir(path)

        file_name = "model_{}-{}.pth".format(
            self.hyper_params["epochs"], time.ctime().replace(" ", "_")
        )
        path = path + config.modelparam.name_study +'_' + file_name

        checkpoints = self.save_checkpoint()

        torch.save(checkpoints, path)
       
        return path

    def restore_checkpoint(self, checkpoints: dict) -> None:
        
        self._start_epoch = checkpoints["epoch"]
        if not isinstance(self._start_epoch, int):
            raise TypeError

        if self._is_parallel:
            self.model.module.load_state_dict(checkpoints["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoints["model_state_dict"])

        self.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
        

    def restore_from_file(self, path: str, map_location: str = "cpu") -> None:

        checkpoints = torch.load(path, map_location=map_location)
        self.restore_checkpoint(checkpoints)