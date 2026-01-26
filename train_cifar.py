from models import Unet
from datasetLoaders import make_cifar10_train_loader
import matplotlib.pyplot as plt
from scripts import train
import torch.nn.functional as F
from ema import EMA
from torchvision.transforms import ToPILImage
from torchvision.datasets import CIFAR10
import shutil, os, tqdm, pickle, torch
from diffusion import ddim_sample



def load_real_cifar(data_dir, N, soft_N):
    soft_real_dir = data_dir + "/real_soft"
    hard_real_dir = data_dir + "/real_hard"
    if os.path.exists(soft_real_dir): shutil.rmtree(soft_real_dir)
    if os.path.exists(hard_real_dir): shutil.rmtree(hard_real_dir)

    
    os.makedirs(soft_real_dir, exist_ok=True)
    os.makedirs(hard_real_dir, exist_ok=True)
    
    ds = CIFAR10(root=data_dir, train=True, download=True)
    for i in tqdm.tqdm(range(N), desc="Saving CIFAR real images", dynamic_ncols=True):
        img, _ = ds[i]
    
        # save all into hard set
        img.save(f"{hard_real_dir}/{i:05d}.png")

        # save first soft_N into soft set
        if i < soft_N:
            img.save(f"{soft_real_dir}/{i:05d}.png")
    
    print("saved soft real images to:", soft_real_dir)
    print("saved hard real images to:", hard_real_dir)


def main():
    
    N = 10000
    soft_N = 5000
    is_windows = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_dir = "/kaggle/working"
    real_dir = "fid"
    data_not_loaded = True
    
    model = Unet(
        in_resolution = 32, 
        input_ch = 3,
        ch = 128,
        output_ch = 3,
        num_res_blocks = 2,
        temb_dim = 256,
        attn_res = set([16]),  
        dropout = 0.1,
        ch_mult=[1,2,2,2]
        )

    model = model.to(device)
    ema = EMA(model, decay=0.9999, device=device)
    
    if is_windows: dataLoader = make_cifar10_train_loader(num_workers=0, pin_memory=False)
    else: dataLoader = make_cifar10_train_loader()
    
    
    
    if data_not_loaded: load_real_cifar(main_dir+"/"+real_dir, N, soft_N)
    
    train(  
            main_dir = main_dir,
            start_step=0,
            real_dir=real_dir,
            sample_dir = "samples",
            model = model, 
            ema=ema, 
            dataLoader=dataLoader,
            optimizer=torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0 ), 
            loss_fn=F.mse_loss,
            exp_no = 1,
            sampling_fn = ddim_sample,
            device = device,
            soft_N=soft_N,
            ckpt_dir="checkpoints",
            save_dir = "saves",
            hard_N=N,
            fid_warmup_steps=5000,
            fid_interval_N=50_000,
            fid_interval_n=10_000
        )
    



    
    
    
    
    
if __name__ == "__main__":
    main()

