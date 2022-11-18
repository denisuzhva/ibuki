import torch

import os
from os.path import exists
import numpy as np
import pandas as pd
import yaml
from neurowhore.data_processing.datasets import SimpleWavHandler
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import nets.sd_net
from trainer import train_model



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Best available device:', device)
    
    cfg_path = f'./prototype2_csnet/cfg/'
    
    # Load general config
    with open(cfg_path + 'general.yaml') as f:
        general_cfg = yaml.safe_load(f)
    
    run_names = general_cfg['runs']
    for run_name in run_names:
        cfg_path_run = cfg_path + run_name + '/'
        with open(cfg_path_run + 'dataset.yaml') as f:
            dataset_params = yaml.safe_load(f)
        with open(cfg_path_run + 'models.yaml') as f:
            all_models_data = yaml.safe_load(f)
        with open(cfg_path_run + 'trainer.yaml') as f:
            trainer_params = yaml.safe_load(f)
            
        # Load data
        sr = dataset_params['sr']
        sample_size = dataset_params['sample_size']
        train_path = dataset_params['train_path']
        valid_path = dataset_params['valid_path']
        mono = dataset_params['mono']
        train_dataset = SimpleWavHandler(train_path, sr, 
                                         mono=mono,
                                         sample_size=sample_size, 
                                         unfolding_step=sample_size//2,
                                         device=device)
        valid_dataset = SimpleWavHandler(valid_path, sr, 
                                         mono=mono,
                                         sample_size=sample_size, 
                                         unfolding_step=sample_size//2,
                                         use_part=0.1,
                                         device=device)
        
        # Define the models
        models = {} 
        
        for model_type, model_data in all_models_data.items():
            mname = model_data['name']
            reqgrad_flag = model_data['reqgrad']
            model_class = getattr(nets.sd_net, mname)
            model = model_class(model_data['params'],)
            model.to(device)
            models[model_type] = (mname, model, reqgrad_flag)
            #if model_type == 'decoder':
            #    model.print_arch_log()
        
        # Prepare data
        batch_size = trainer_params['batch_size']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            
        train_log_dir = './prototype2_csnet/train_logs/'
        os.makedirs(train_log_dir, exist_ok=True)
        train_log_path = train_log_dir + run_name + '_log.csv'

        trained_dump_dir = './prototype2_csnet/checkpoints/'
        os.makedirs(trained_dump_dir, exist_ok=True)
        opt_path = trained_dump_dir + run_name + '_opt.pth'
        
        # Check if model and optimizer dump exists
        # TODO make it work
        if exists(opt_path):
            print("Models and opt loaded")
            opt_chkp = torch.load(opt_path)
            for mname, model, _ in models.values():
                model.load_state_dict(torch.load(trained_dump_dir + f'{run_name}_{mname}.pth'), strict=False)
        else:
            print("Models and opt not found! Initiating new training instance")
            opt_chkp = None
        
        # Check if log exists
        if os.path.exists(train_log_path):
            print("Log loaded")
            log_df = pd.read_csv(train_log_path)
            last_epoch = log_df['epoch'].iloc[-1]
            min_v_loss = log_df['min_v_loss'].iloc[-1]
        else:
            print("Initiating new log")
            last_epoch = 0
            min_v_loss = np.Inf

        # Train it
        if trainer_params['do_train']:
            train_model(
                train_dataloader,
                valid_dataloader,
                models,
                trainer_params['model_handler'],
                n_epochs=trainer_params['epochs'],
                learning_rate_params=trainer_params['lr_params'],
                crit_lambdas=trainer_params['losses'],
                run_name=run_name,
                device=device,
                log_df_path=train_log_path,
                trained_dump_dir=trained_dump_dir,
                opt_path=opt_path,
                last_epoch=last_epoch,
                min_v_loss=min_v_loss,
                opt_chkp=opt_chkp,
            )
