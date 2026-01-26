from scripts import evaluate_fid, save_loss_fid_plots
from ema import EMA
from models import Unet
from diffusion import ddim_sample
import torch




def main():
    main_dir = "/kaggle/working"
    exp_no = "1"
    N = 10000
    soft_N = 5000

    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_point_path = main_dir+f"/{exp_no}/checkpoints"
    results_save_path = main_dir+f"/{exp_no}/saves"
    
    ckpt = torch.load(check_point_path)
    
    
    
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
    # model.load_state_dict(ckpt["model"])
    # model = model.to(device)
    
    # ema = EMA(model, decay=0.9999, device=device)
    # ema.ema.load_state_dict(ckpt["ema"])
    
    # fid = evaluate_fid(
    #             ema = ema.ema,
    #             sampling_fn = ddim_sample,
    #             hard_N = N,
    #             fdir1 = main_dir + "/fid",
    #             fdir2 = main_dir + "/samples",
    #             device = device,
    #             batch_size = 256,
    #             steps = 50,
    #             eta = 0.0,
    #             image_size = 32,
    #             )
    save_loss_fid_plots(
                check_point_path,
                results_save_path,
                )

if __name__ == "__main__":
    main()


