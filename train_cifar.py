from models import Unet
from datasetLoaders import make_cifar10_train_loader
import matplotlib.pyplot as plt
from scripts import train, model_load_latest_state
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
    
    
    #settings
    is_windows = False
    N = 10_000
    soft_N = 5_000
    fid_warmup_steps=5_000
    fid_interval_N=50_000
    fid_interval_n=10_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main_dir = "/scratch/aalefern/Building_Diffusion_Models/working"
    real_dir = "fid"
    ckpt_dir = "checkpoints"
    sample_dir = "samples"
    save_dir = "saves"
    exp_no = 1
    
    
   
    # create and load model and ema of the model
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=0.0 )
    loss_fn = F.mse_loss
    
    # Load latest model if saved model exists
    start_step = 0
    loss_list = fid_list = []
    best_fid = float('inf')
    
    if os.path.exists(main_dir+f"/{exp_no}/"+ckpt_dir):
        start_step, best_fid, loss_list, fid_list = model_load_latest_state(
                                                                            exp_no = exp_no,
                                                                            main_dir = main_dir, 
                                                                            ckpt_dir=ckpt_dir, 
                                                                            model = model, 
                                                                            optimizer = optimizer, 
                                                                            ema = ema, 
                                                                            device = device
                                                                            )
        
        print(f"Resumed from step {start_step}, best FID {best_fid:.4f}")
    else:
        print("No checkpoint found. Starting from scratch.")
    
    if is_windows: dataLoader = make_cifar10_train_loader(num_workers=0, pin_memory=False)
    else: dataLoader = make_cifar10_train_loader()
    
    
    #Download the real data for FID calculation
    if not os.path.exists(main_dir+"/"+real_dir): 
        print("Cifar real data does not exist. Downloading...")
        load_real_cifar(main_dir+"/"+real_dir, N, soft_N)
    
    
    
    #The fun part: Training
    
    train(  
            main_dir = main_dir,
            start_step=start_step,
            real_dir=real_dir,
            sample_dir = sample_dir,
            model = model, 
            ema=ema, 
            dataLoader=dataLoader,
            optimizer=optimizer, 
            loss_fn=loss_fn,
            exp_no = exp_no,
            sampling_fn = ddim_sample,
            device = device,
            soft_N=soft_N,
            ckpt_dir=ckpt_dir,
            save_dir = save_dir,
            hard_N=N,
            fid_warmup_steps=fid_warmup_steps,
            fid_interval_N=fid_interval_N,
            fid_interval_n=fid_interval_n,
            loss_list = loss_list,
            fid_list = fid_list,
            best_fid = best_fid,
            
        )
    



    
    
    
    
    
if __name__ == "__main__":
    main()

