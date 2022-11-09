import torch

import os
import yaml
from datasets import SimpleWavHandler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nets.noise_sched import NoiseScheduler
from nets.wavenet_unet import UWUNet_v1
from trainer import train_sd_model



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_name = 'UWUNet_experiment_1'
    writer = SummaryWriter(f'runs/{run_name}/')
    cfg_path = f'./cfg/{run_name}/'
    
    with open(cfg_path + 'dataset.yaml') as f:
        dataset_params = yaml.safe_load(f)
    with open(cfg_path + 'model.yaml') as f:
        model_params = yaml.safe_load(f)
    with open(cfg_path + 'trainer.yaml') as f:
        trainer_params = yaml.safe_load(f)
        
    # Load data
    sr = dataset_params['sr']
    sample_size = dataset_params['sample_size']
    train_path = dataset_params['train_path']
    test_path = dataset_params['test_path']
    train_dataset = SimpleWavHandler(train_path, sr, 
                                     mono=True if model_params['wav_channels'] == 1 else False,
                                     sample_size=sample_size, 
                                     unfolding_step=sample_size//2,
                                     device=device)
    test_dataset = SimpleWavHandler(test_path, sr, 
                                    mono=True if model_params['wav_channels'] == 1 else False,
                                    sample_size=sample_size, 
                                    unfolding_step=sample_size//2,
                                    use_part=0.1,
                                    device=device)

    # Forward diffusion: Noise Scheduler
    t_max = model_params['t_max']
    noise_sched = NoiseScheduler(t_max, 
                                 end=model_params['beta_end'], 
                                 distrib_type=model_params['distrib_type']).to(device)
    
    # Backward diffusion: UWUNet
    denoiser_model = UWUNet_v1(wav_channels=model_params['wav_channels'],
                               down_channels=model_params['down_channels'],
                               up_channels=model_params['up_channels'],
                               time_emb_dim=model_params['time_emb_dim'],
                               wn_dilation_depth=model_params['wn_dilation_depth'],
                               wn_repeats=model_params['wn_repeats']).to(device)
    
    # Train the model 
    batch_size = trainer_params['batch_size']
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
    train_log_dir = './train_logs/'
    train_log_path = train_log_dir + run_name + '_log.csv'
    checkpoint_dir = './checkpoints/'
    model_checkpoint_path = checkpoint_dir + run_name + '_model.pth'
    opt_checkpoint_path = checkpoint_dir + run_name + '_opt.pth'
    os.makedirs(train_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if trainer_params['do_train']:
        train_sd_model(
            train_dataloader,
            test_dataloader,
            noise_sched,
            denoiser_model,
            batch_size,
            t_max,
            n_epochs=trainer_params['epochs'],
            learning_rate=trainer_params['lr'],
            crit_lambdas=trainer_params['losses'],
            device=device,
            log_df_path=train_log_path,
            model_dump_path=model_checkpoint_path,
            chkpnt_dump_path=opt_checkpoint_path,
        )
        
    #writer.add_graph(denoiser_model, (next(iter(train_dataloader)), t))
    #writer.close()