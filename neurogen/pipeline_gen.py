import torch

import os
from os.path import exists
import numpy as np
import pandas as pd
import yaml
from data_processing.datasets import SimpleDilatedWavHandler
from torch.utils.data import DataLoader

import nets.gen_net
from trainer import train_model
from utils import count_parameters



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Best available device:', device)
    
    cfg_path = f'./neurogen/cfg/'
    
    # Load general config
    with open(cfg_path + 'general.yaml') as f:
        general_cfg = yaml.safe_load(f)
    
    run_names = general_cfg['runs']
    for run_name in run_names:
        print(f"Run: {run_name}")
        cfg_path_run = cfg_path + run_name + '/'
        with open(cfg_path_run + 'dataset.yaml') as f:
            dataset_params = yaml.safe_load(f)
        with open(cfg_path_run + 'models.yaml') as f:
            all_models_data = yaml.safe_load(f)
        with open(cfg_path_run + 'trainer.yaml') as f:
            trainer_params = yaml.safe_load(f)
            
        # Load data
        mono = dataset_params['mono']
        sr = dataset_params['sr']
        patch_size = dataset_params['patch_size']
        n_patches = dataset_params['n_patches']
        dilation_depth = dataset_params['dilation_depth']
        train_use_part = dataset_params['train_use_part']
        valid_use_part = dataset_params['valid_use_part']
        train_path = dataset_params['train_path']
        valid_path = dataset_params['valid_path']
        train_dataset = SimpleDilatedWavHandler(train_path, sr, 
                                                mono=mono,
                                                patch_size=patch_size,
                                                n_patches=n_patches,
                                                dilation_depth=dilation_depth,
                                                use_part=train_use_part)
        valid_dataset = SimpleDilatedWavHandler(valid_path, sr, 
                                                mono=mono,
                                                patch_size=patch_size,
                                                n_patches=n_patches,
                                                dilation_depth=dilation_depth,
                                                use_part=valid_use_part)
        _, context_size = train_dataset.get_context()
        print(f"WAV data context {context_size} sec at {sr} kHz")
        
        # Define the models
        models = {} 
        n_parameters = {}
        
        for model_type, model_data in all_models_data.items():
            mname = model_data['name']
            reqgrad_flag = model_data['reqgrad']
            if not model_data['params']['d_model']:
                model_data['params']['d_model'] = patch_size
            if not model_data['params']['dilation_depth']:
                model_data['params']['dilation_depth'] = dilation_depth
            model_class = getattr(nets.gen_net, mname)
            model = model_class(model_data['params'],)
            model.to(device)
            models[model_type] = (mname, model, reqgrad_flag)
            n_parameters[model_type] = count_parameters(model) / 1e6
            #if model_type == 'decoder':
            #    model.print_arch_log()
        print("# parameters, M: ", n_parameters)
        
        # Prepare data
        batch_size = trainer_params['batch_size']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            
        train_log_dir = './neurogen/train_logs/'
        os.makedirs(train_log_dir, exist_ok=True)
        train_log_path = train_log_dir + run_name + '_log.csv'

        trained_dump_dir = './neurogen/checkpoints/'
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
            print("Models and opt dumps NOT found! Initiating new training instance")
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
        trainer_params['model_handler']['params']['n_classes'] = \
            all_models_data['transformer']['params']['n_q_out']
        if trainer_params['do_train']:
            train_model(
                train_dataloader,
                valid_dataloader,
                models,
                model_handler_data=trainer_params['model_handler'],
                n_epochs=trainer_params['epochs'],
                learning_rate_params=trainer_params['lr_params'],
                crit_lambdas=trainer_params['losses'],
                run_name=run_name,
                device=device,
                log_df_path=train_log_path,
                trained_dump_dir=trained_dump_dir,
                opt_path=opt_path,
                validate_each_n_epoch=trainer_params['validate_each_epoch'],
                last_epoch=last_epoch,
                min_v_loss=min_v_loss,
                opt_chkp=opt_chkp,
            )
