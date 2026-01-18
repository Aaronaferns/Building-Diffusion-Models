

import os

import torch
import torch.nn.functional as F
from itertools import cycle
from diffusion import diffusionTrainStep, sample

from tqdm import tqdm
import matplotlib.pyplot as plt





def saveCheckPoint(save_folder, exp_no, step, model, optimizer, loss_list):
    save_dir = save_folder
    exp_dir = os.path.join(save_dir, str(exp_no))
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_path = os.path.join(exp_dir, f"{step}_checkpoint.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "losses": loss_list,
    }, ckpt_path)



def train(model, ema, dataLoader, optimizer, loss_fn, num_steps, save_interval, save_folder, exp_no, device):
    model.train()
    data_iter = cycle(dataLoader)
    pbar = tqdm(range(num_steps), desc="Training")
    loss_list = []
    for step in pbar:
        X0, _ = next(data_iter)
        loss = diffusionTrainStep(model, ema, optimizer, loss_fn, X0, device)
        loss_list.append(loss)
        if (step > 0 and step%save_interval == 0 ) or step == num_steps-1:
            saveCheckPoint(
                save_folder=save_folder,
                exp_no = exp_no,
                step = step,
                model = model,
                optimizer = optimizer,
                loss_list = loss_list
                )
    
    print("Done Training")
    return loss_list
    

