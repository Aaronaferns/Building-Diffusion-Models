from scripts import evaluate_fid, save_loss_fid_plots
from ema import EMA
from models import Unet
from diffusion import ddim_sample
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt



def main():
    main_dir = "/scratch/aalefern/Building_Diffusion_Models/working"
    exp_no = "1"
    N = 10000
    soft_N = 5000

    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_point_path = main_dir+f"/{exp_no}/checkpoints"
    results_save_path = main_dir+f"/{exp_no}/saves"
    
    # ckpt = torch.load(check_point_path)
    # print(ckpt["step"])
    
    # model = Unet(
    #     in_resolution = 32, 
    #     input_ch = 3,
    #     ch = 128,
    #     output_ch = 3,
    #     num_res_blocks = 2,
    #     temb_dim = 256,
    #     attn_res = set([16]),  
    #     dropout = 0.1,
    #     ch_mult=[1,2,2,2]
    #     )

    # model = model.to(device)
    # ema = EMA(model, decay=0.9999, device=device)
    # model.load_state_dict(ckpt["model"])
    # ema.ema.load_state_dict(ckpt["ema"])
    # print("successfully loaded model")
    # save_loss_fid_plots(
    #             check_point_path,
    #             results_save_path,
    #             )


    # Paths
    IMAGE_DIR = "/scratch/aalefern/Building_Diffusion_Models/working/1/samples"
    OUTPUT_PATH = results_save_path + "/image_grid.png"

    # Collect image files
    image_files = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])[:25]

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    axes = axes.flatten()

    for ax, img_path in zip(axes, image_files):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(image_files):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved image grid to {OUTPUT_PATH}")



if __name__ == "__main__":
    main()


