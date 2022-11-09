# trainer.py

import os
import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.optim import Adam



def init_weights_xavier(m):
    """Xavier weight initializer."""
    if type(m) == (nn.Conv2d or nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_kaiming(m):
    """Kaiming weight initializer."""
    if type(m) == (nn.Conv2d or nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def fft_l1_norm(fft_size=4096):
    def get_norm(data, rec_data):
        l = data.shape[-1]
        pad_size = fft_size - l
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        data_padded = nn.functional.pad(data, (pad_left, pad_right))
        data_fft = torch.abs(torch.fft.fft(data_padded))
        norm_value = nn.functional.smooth_l1_loss(data_fft, torch.zeros_like(data_fft).to(data_fft.device))
        return norm_value
    return get_norm


def train_sd_model(train_loader, valid_loader, 
                   noise_sched,
                   denoiser_model, 
                   batch_size, 
                   t_max,
                   n_epochs, learning_rate, 
                   crit_lambdas, 
                   device, 
                   log_df_path, model_dump_path, chkpnt_dump_path,
                   validate_each_n_epoch=5, last_epoch=0, 
                   min_v_loss=np.Inf,
                   checkpoint=None):
    """
    The trainer function for a compressed sensing network training and validation.

        Parameters
        ----------
        dataset :               A dataset to get loaders from
        noise_sched :           Noise scheduler
        denoiser_model :        Denoiser model
        batch_size :            Size of a batch
        n_epochs :              Number of epochs
        learning_rate :         Learning rate
        crit_lambdas :          Weight coefficients for loss functions
        device :                Current device (cuda or cpu)
        log_df_path :           Path to training logs
        model_dump_path :       Path to model dump
        chkpnt_dump_path :      Path to the optimizer and scheduler checkpoints
        validate_each_n_epoch : An interval between epochs with validation performed
        last_epoch :            Last epoch before current transfer learning
        min_v_loss :            Minimum validation loss among all epochs
        checkpoint :            Optimizer and scheduler checkpoints
    """

    n_train_batches = len(train_loader) 
    n_valid_batches = len(valid_loader) 

    # Loss and optimizer
    crits = {
        "l2": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "l1S": nn.SmoothL1Loss(),
        "l1norm": fft_l1_norm(),
    }             
    loss_vals = {
        "t": {},
        "v": {}
    }
    for mode in loss_vals.keys():
        for lm in crit_lambdas.keys():
            loss_vals[mode][lm] = 0.
    
    model_params = denoiser_model.parameters()
    optimizer = Adam(model_params, lr=learning_rate)
    #optimizer = SGD(model_params, lr=learning_rate)
    #optimizer = Adagrad(model_params, lr=learning_rate)
    gamma = 0.5
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=gamma)
    denoiser_model.apply(init_weights_xavier)

    # Checkpoint processing
    optimizer_chkpt_name = 'optimizer'
    scheduler_chkpt_name = 'scheduler'
    if checkpoint:
        print("Optimizer and Scheduler restored")
        optimizer.load_state_dict(checkpoint[optimizer_chkpt_name])
        lr_scheduler.load_state_dict(checkpoint[scheduler_chkpt_name])

    # Training iterations
    if last_epoch == 0:
        log_header = True
    else:
        log_header = False
    start_t = time.time() 
    for epoch in range(last_epoch+1, n_epochs+1):

        if epoch % validate_each_n_epoch == 0:
            do_eval = True
        else:
            do_eval = False

        for key in loss_vals.keys():
            for lm in crit_lambdas.keys():
                loss_vals[key][lm] = 0.
            
        # Training
        denoiser_model.train()
        for batch_idx, data in enumerate(train_loader):
            data = data.float().to(device)
            t = torch.randint(0, t_max, (batch_size,), device=device).long()
            data_noised, noise = noise_sched(data, t)
            pred_noise = denoiser_model(data_noised, t)

            train_losses = {}
            for lm in crit_lambdas.keys():
                train_losses[lm] = crit_lambdas[lm] * crits[lm](noise, pred_noise)
                loss_vals['t'][lm] += train_losses[lm].cpu().item() / n_train_batches

            optimizer.zero_grad()
            train_loss = sum(train_losses.values())
            train_loss.backward()
            optimizer.step()
        
        lr_scheduler.step()

        if do_eval:
            ## Validation
            denoiser_model.eval()
            for batch_idx, data in enumerate(valid_loader):
                data = data.float().to(device)
                t = torch.randint(0, t_max, (batch_size,), device=device).long()
                data_noised, noise = noise_sched(data, t)
                pred_noise = denoiser_model(data_noised, t)
                valid_losses = {}
                for lm in crit_lambdas.keys():
                    valid_losses[lm] = crit_lambdas[lm] * crits[lm](noise, pred_noise)
                    loss_vals['v'][lm] += valid_losses[lm].cpu().item() / n_valid_batches

            if loss_vals['v'][list(crit_lambdas.keys())[0]] < min_v_loss:
                torch.save(denoiser_model, model_dump_path)
                torch.save({
                    optimizer_chkpt_name: optimizer.state_dict(),
                    scheduler_chkpt_name: lr_scheduler.state_dict()
                }, chkpnt_dump_path)
                min_v_loss = loss_vals['v'][list(crit_lambdas.keys())[0]]

            d = {"epoch": [epoch], "min_v_loss": [min_v_loss]}
            for mode in loss_vals.keys():
                for lm in crit_lambdas.keys():
                    d[mode + "_" + lm] = [loss_vals[mode][lm]]
                    loss_vals[mode][lm] = 0. 
            d_rounded = {key: round(value[0], 7) for key, value in d.items()}
            print(d_rounded)
            df = pd.DataFrame.from_dict(d)
            df.to_csv(log_df_path, mode='a', header=log_header, index=False)
            log_header = False

    print("t: ", time.time() - start_t)




