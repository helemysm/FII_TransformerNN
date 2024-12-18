import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os 

gray_color_with_opacity = 'rgba(169, 169, 169, 0.7)'  # Adjust the alpha channel (last value) for opacity

class WandbLogger:
    def __init__(self, project_name, api_key):
        wandb.init(project=project_name)#, api_key=api_key)
        
    def log_metric(self, metric_name, value, step=None):
        wandb.log({metric_name: value}, step=step)
    
    def log_metrics_dict(self, value, step=None):
        wandb.log(value)
    
    def log_figure(self, figure, step=None):
        wandb.log(figure)
    
    def log_lightcurve_plot(self, ts_tensor_batch, ts_tensor_time_batch, ts_tensor_predicted_labels, ts_tensor_real_labels, epoch=None, step=None):
        
        for i in range(min(len(ts_tensor_batch), 5)): 
            plt.figure(figsize=(10, 7))
            plt.plot(ts_tensor_batch[i].cpu().numpy(), label='Time Series')#, s=1.7)
            plt.title(f"Light curve - Predicted Label: {ts_tensor_predicted_labels[i].cpu().numpy()}")
            plt.xlabel('Time')
            plt.ylabel('Flux')
            plt.legend()
            wandb.log({f"Light curve {i + 1} (Epoch {epoch + 1})": wandb.Image(plt)})
            plt.close()


    def log_lightcurve_scatter(self, ts_tensor_batch, ts_tensor_time_batch, ts_tensor_predicted_labels, ts_tensor_real_labels, epoch=None, step=None):
        
        for i in range(min(len(ts_tensor_batch), 5)): 
            fig = go.Figure()
            #start_date = "2023-01-01 00:00:00"
            #start_datetime = np.datetime64(start_date)
            #data_points = np.random.randn(1000)  
            #time_values = np.arange(start_datetime, start_datetime + np.timedelta64(1000, 'h'), dtype='datetime64[h]')
            #fig.add_trace(go.Scatter(x=time_values, y=data_points, mode='lines')) #name='After
            ts_time = ts_tensor_time_batch[i].squeeze().cpu().numpy().tolist()
            ts_flux = ts_tensor_batch[i].squeeze().cpu().numpy()
            
            fig.add_trace(go.Scatter(x=ts_time, y=ts_flux, mode='lines', line=dict(color=gray_color_with_opacity, width=2))) #name='After (positive)'
            fig.update_layout(title=f'Light curve {i} -Epoch:{epoch} - Step:{step} - Label: {ts_tensor_real_labels[i].cpu().numpy()}', xaxis_title='Time', yaxis_title='Flux')
            #{summary_name: wandb.Plotly(figure)},
            #wandb.log({f"light_curve_{i}": wandb.Plotly(fig)})
            wandb.log({f"light_curve_{i}_epoch{epoch}_step_{step}": fig})

        

    def log_parameters(self, config):
        
        model_params = {"layers": config.modelparam.layers, 
              "n_heads": config.modelparam.n_heads,
              "emb_dim": config.modelparam.emb_dim,
              "factor" : config.modelparam.factor,
              "dropout_rate": config.modelparam.dropout_rate,
              "num_random": config.modelparam.num_random,
              "window_size": config.modelparam.window_size,
              "pooling_layer": config.modelparam.pooling_layer,
              "lr": config.hyperparameters.lr}

        hyperparameters = {
            "learning_rate": 0.001,
            "batch_size": 100
        }

        wandb.config.update(model_params)

    def log_model(self, model, log_freq=1000):
        #wandb.watch(model, log=log_freq)
        wandb.watch(model, log='all')
        
        
    def set_name(self, name):
        wandb.run.name = name
        #wandb.set_name(name)
        
    def get_url_wb(self):
        run_url = wandb.run.url
        return run_url
        
    def finish_wb(self):
        wandb.finish()