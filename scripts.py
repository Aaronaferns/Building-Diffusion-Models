
import pickle
import os
from models import Unet
import torch
import torch.nn.functional as F
from itertools import cycle
from diffusion import diffusionTrainStep, sample
from datasetLoaders import make_cifar10_train_loader
from tqdm import tqdm
import matplotlib.pyplot as plt




device = device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

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



def train(model, dataLoader, optimizer, loss_fn, num_steps, save_interval, save_folder, exp_no):
    model.train()
    data_iter = cycle(dataLoader)
    pbar = tqdm(range(num_steps), desc="Training")
    loss_list = []
    for step in pbar:
        X0, _ = next(data_iter)
        loss = diffusionTrainStep(model, optimizer, loss_fn, X0, device)
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
    

def main():
    model = Unet(
        in_resolution = 32,  # CIFAR-10 images are 32x32
        input_ch = 3,
        ch = 128,
        output_ch = 3,
        num_res_blocks = 3,
        temb_dim = 256,
        attn_res = set([16]),  # Attention at lower resolutions for 32x32
        dropout = 0.1,
        resam_with_conv=True,
        ch_mult=[1,2,4,8]
        )

    model = model.to(device)
    dataLoader = make_cifar10_train_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0 )
    loss_fn = F.mse_loss
    NUM_TRAIN_STEPS = 1000
    exp_no = 1
    save_interval = 100
    save_folder = "saves"
    loss_list = train(model, dataLoader, optimizer, loss_fn, NUM_TRAIN_STEPS, save_interval, save_folder, exp_no)
    
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("DDPM Training Loss")
    plt.savefig(f"{exp_no}_loss_curve.png", dpi=200, bbox_inches="tight")
    plt.show()


    
    
    
    
    
if __name__ == "__main__":
    main()

