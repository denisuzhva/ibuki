import os
import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.optim import Adam, SGD
from utils import init_weights_xavier, fft_l1_norm
from utils import wav16_to_onehot



def sd_wavenet_handler(models, data, params, device):
    batch_size = data.shape[0]
    data = data.to(device)
    t = torch.randint(0, params['t_max'], (batch_size,), device=data.get_device()).long()
    noise_sched = models['noise_sched'][1]
    denoiser = models['denoiser'][1]
    data_noised, noise = noise_sched(data, t)
    pred_noise = denoiser(data_noised, t)
    return pred_noise, noise


def dtf_gen_handler(models, data, params, device):
    in_seq, target = data
    in_seq = in_seq.float().to(device) / 2**15
    target = target.to(device)
    n_classes = params['n_classes']
    dtf_generator = models['transformer'][1]
    out_seq = dtf_generator(in_seq)
    out_sample = out_seq[-1]
    target_oh = wav16_to_onehot(target, n_classes, do_mu=True).float()
    return out_sample, target_oh


def train_model(train_loader, valid_loader, 
                models, 
                model_handler_data,
                n_epochs,  
                learning_rate_params,
                crit_lambdas, 
                device, 
                run_name,
                log_df_path, trained_dump_dir, opt_path,
                validate_each_n_epoch=5, last_epoch=0, 
                min_v_loss=np.Inf,
                opt_chkp=None):
    """
    The trainer function for a compressed sensing network training and validation.

        Parameters
        ----------
        train_loader :          Train dataset loader
        valid_loader :          Validation dataset loader
        models :                Dictionary with model blocks; each tagget if requires gradients or not
        model_handler_name :    An algorithm of model handling 
        n_epochs :              Number of epochs
        learning_rate_params :  Learning rate params for optimizer and scheduler
        crit_lambdas :          Weight coefficients for loss functions
        device :                Current device (cuda or cpu)
        run_name :              Name of the experiment for logging
        log_df_path :           Path to training logs
        trained_dump_path :     Path to model dump
        opt_path :              Path to the optimizer and scheduler checkpoints
        validate_each_n_epoch : An interval between epochs with validation performed
        last_epoch :            Last epoch before current transfer learning
        min_v_loss :            Minimum validation loss among all epochs
        opt_chkp :              Optimizer and scheduler checkpoints
    """

    n_train_batches = len(train_loader) 
    print(f"# train batches: {n_train_batches}")
    n_valid_batches = len(valid_loader) 
    
    # Model handlers
    model_handlers_all = {
        'sd_wavenet': sd_wavenet_handler,
        'dtf_gen': dtf_gen_handler,
    }
    model_handler = model_handlers_all[model_handler_data['name']]

    # Loss and optimizer
    crits = {
        'l2': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'l1S': nn.SmoothL1Loss(),
        'l1norm': fft_l1_norm(),
        'ce': nn.CrossEntropyLoss(label_smoothing = 0.1),
    }             
    loss_vals = {
        't': {},
        'v': {}
    }
    for mode in loss_vals.keys():
        for lm in crit_lambdas.keys():
            loss_vals[mode][lm] = 0.
    
    model_params = []
    for mname, model, reqgrad_flag in models.values():
        model_params += list(model.parameters())
        if reqgrad_flag:
            model.apply(init_weights_xavier)
    optimizer = Adam(model_params, lr=learning_rate_params['lr'], betas=(0.9, 0.98), eps=1.0e-9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, 
                                                   step_size=learning_rate_params['sched_step'], 
                                                   gamma=learning_rate_params['sched_gamma'])

    # Checkpoint processing
    optimizer_chkpt_name = 'optimizer'
    scheduler_chkpt_name = 'scheduler'
    if opt_chkp:
        print("Optimizer and Scheduler restored")
        optimizer.load_state_dict(opt_chkp[optimizer_chkpt_name])
        lr_scheduler.load_state_dict(opt_chkp[scheduler_chkpt_name])

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
        for _, model, reqgrad_flag in models.values():
            if reqgrad_flag:
                model.train()
            else:
                model.eval()
        for batch_idx, data in enumerate(train_loader):
            #data = data.float().to(device)
            #if batch_idx % 1000 == 0:
            #    print(f"batch # {batch_idx}")
            pred, target = model_handler(models, 
                                         data,
                                         model_handler_data['params'],
                                         device,)
            train_losses = {}
            for lm in crit_lambdas.keys():
                train_losses[lm] = crit_lambdas[lm] * crits[lm](pred, target)
                loss_vals['t'][lm] += train_losses[lm].cpu().item() / n_train_batches

            print(f"epoch: {epoch}, batch: {batch_idx}, train_losses: {train_losses}")
            optimizer.zero_grad()
            train_loss = sum(train_losses.values())
            train_loss.backward()
            optimizer.step()
        
        lr_scheduler.step()

        if do_eval:
            ## Validation
            for _, model, _ in models.values():
                model.eval()
            for batch_idx, data in enumerate(valid_loader):
                #data = data.float().to(device)
                pred, target = model_handler(models, 
                                             data, 
                                             model_handler_data['params'],
                                             device,)
                valid_losses = {}
                for lm in crit_lambdas.keys():
                    valid_losses[lm] = crit_lambdas[lm] * crits[lm](pred, target)
                    loss_vals['v'][lm] += valid_losses[lm].cpu().item() / n_valid_batches

            if loss_vals['v'][list(crit_lambdas.keys())[0]] < min_v_loss:
                for mname, model, _ in models.values():
                    torch.save(model.state_dict(), trained_dump_dir + f'{run_name}_{mname}.pth')
                torch.save({
                    optimizer_chkpt_name: optimizer.state_dict(),
                    scheduler_chkpt_name: lr_scheduler.state_dict()
                }, opt_path)
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





