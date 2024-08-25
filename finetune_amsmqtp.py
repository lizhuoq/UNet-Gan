import argparse
import os

import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from data_utils import AMSMQTPDataset
from loss import GeneratorLoss
from model import UNetGenerator, Discriminator, MBConvUNetGenerator
from transformers import get_cosine_schedule_with_warmup
from utils.tools import EarlyStopping, load_partial_state_dict, iterate_batches
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
import pandas as pd
from sklearn.model_selection import KFold
import xarray as xr

# 过滤所有的 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

model_map = {
    "UNet": UNetGenerator, 
    "MBConvUNet": MBConvUNetGenerator
}

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')
parser.add_argument('--generator', type=str)
parser.add_argument('--in_channel', required=True, type=int)
parser.add_argument('--out_channel', required=True, type=int)
parser.add_argument('--G_learning_rate', type=float)
parser.add_argument('--D_learning_rate', type=float)
parser.add_argument('--patience', required=True, type=int)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--test', default=False, action="store_true")
parser.add_argument('--inference', default=False, action="store_true")
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--warmup_epochs', required=True, type=int)
parser.add_argument("--seed", default=2024, type=int)
parser.add_argument('--perceptual_loss', type=str, default="vgg16", choices=["vgg16", "vgg19"])

parser.add_argument('--kernel_size', type=int, default=3, choices=[3, 5])

# MBConv
parser.add_argument('--exp_ratio', type=int, default=2, choices=[2, 4])
parser.add_argument('--squeeze_ratio', type=int, default=2, choices=[2, 4])

parser.add_argument('--fold_n', type=int, choices=[1, 2, 3, 4, 5])
parser.add_argument('--point_alpha', type=float)

# inference
parser.add_argument('--exp_id', type=int, choices=list(range(4)))
parser.add_argument('--history', default=False, action="store_true")


if __name__ == '__main__':
    opt = parser.parse_args()

    I = 0
    if opt.test:
        I = 1
    if opt.inference:
        I = 2

    if opt.debug:
        breakpoint()
        torch.autograd.set_detect_anomaly(True)

    setting = [
        "finetune", 
        f"Generator_{opt.generator}", 
        f"epochs_{opt.num_epochs}", 
        f"Glr_{opt.G_learning_rate}", 
        f"Dlr_{opt.D_learning_rate}", 
        f"patience_{opt.patience}", 
        f"batch_size_{opt.batch_size}", 
        f"warmup_epochs_{opt.warmup_epochs}", 
        f"perceptual_loss_{opt.perceptual_loss}", 
        f"kernel_size_{opt.kernel_size}", 
        f"exp_ratio_{opt.exp_ratio}", 
        f"squeeze_ratio_{opt.squeeze_ratio}", 
        f"point_alpha_{opt.point_alpha}", 
        f"fold_n_{opt.fold_n}"
    ]
    setting = "_".join(setting)
    NUM_EPOCHS = opt.num_epochs
    SEED = opt.seed
    
    data_set = AMSMQTPDataset()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_set)):
        if fold + 1 == opt.fold_n:
            break
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset=data_set, num_workers=opt.num_workers, batch_size=opt.batch_size, 
                              sampler=train_sampler)
    val_loader = DataLoader(dataset=data_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=False, 
                            sampler=val_sampler)
    
    test_loader = val_loader

    Generator = model_map[opt.generator]
    
    netG = Generator(opt).float()
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator(in_channel=opt.out_channel).float()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    print(f"Fold: {opt.fold_n}")
    
    generator_criterion = GeneratorLoss(opt)
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()

    load_partial_state_dict(netG, torch.load("/data/home/scv7343/run/SRGAN-master/training_results/Generator_UNet_epochs_200_Glr_0.0005_Dlr_1e-06_patience_10_batch_size_100_warmup_epochs_0_perceptual_loss_vgg16_kernel_size_3_exp_ratio_2_squeeze_ratio_2/checkpoint.pth"))
    
    optimizerG = optim.Adam(netG.parameters(), lr=opt.G_learning_rate)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.D_learning_rate)

    D_criterion = torch.nn.BCEWithLogitsLoss()
    point_criterion = torch.nn.MSELoss(reduction="none")

    train_sets = len(train_loader)
    schedulerG = get_cosine_schedule_with_warmup(optimizerG, train_sets * opt.warmup_epochs, train_sets * NUM_EPOCHS)
    schedulerD= get_cosine_schedule_with_warmup(optimizerD, train_sets * opt.warmup_epochs, train_sets * NUM_EPOCHS)
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'val_rmse': []}

    early_stopping = EarlyStopping(patience=opt.patience, verbose=True)
    if I == 0:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_bar = tqdm(train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        
            netG.train()
            netD.train()
            for i, (data, target, point_val, point_weight, bound) in enumerate(train_bar):
                point_val = point_val.float().cuda()
                point_weight = point_weight.float().cuda()
                bound = bound.float().cuda()

                bound = bound.unsqueeze(1).repeat(1, target.shape[1], 1, 1)

                g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size

                ############################
                # (1) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                if torch.cuda.is_available():
                    real_img = target.float().cuda()
                if torch.cuda.is_available():
                    z = data.float().cuda()
                fake_img = netG(z)
                fake_out = netD(fake_img)
                real_out = netD(real_img).detach()

                optimizerG.zero_grad()
                g_loss = generator_criterion(fake_out, real_out, fake_img, real_img, bound)

                point_loss = point_criterion(fake_img, real_img)
                point_loss = (point_loss * point_weight)[bound == 1]
                point_loss = point_loss.mean()
                
                g_loss = g_loss + point_loss * opt.point_alpha

                g_loss.backward()
                optimizerG.step()
                schedulerG.step()

                ############################
                # (2) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                fake_out = netD(fake_img.detach())
                real_out = netD(real_img)

                valid = torch.ones((fake_out.shape[0], 1)).to(fake_out.device)
                fake = torch.zeros((fake_out.shape[0], 1)).to(fake_out.device)

                optimizerD.zero_grad()
                loss_real = D_criterion(real_out - fake_out.mean(), valid)
                loss_fake = D_criterion(fake_out - real_out.mean(), fake)
                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                optimizerD.step()
                schedulerD.step()

                # loss for current batch before optimization 
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.mean().item() * batch_size
                running_results['g_score'] += fake_out.mean().item() * batch_size
        
                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
        
            netG.eval()
            out_path = 'training_results/' + setting
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                preds = []
                trues = []
                for val_lr, val_hr, _, _, bound in val_bar:
                    bound = bound.unsqueeze(1).repeat(1, val_hr.shape[1], 1, 1)
                    batch_size = val_lr.size(0)
                    lr = val_lr
                    hr = val_hr
                    if torch.cuda.is_available():
                        lr = lr.float().cuda()
                        hr = hr.float().cuda()
                    sr = netG(lr)

                    sr = sr.detach().cpu().numpy() # b, c, h, w
                    hr = hr.detach().cpu().numpy()
                    bound = bound.numpy()
            
                    preds.append(sr[bound == 1])
                    trues.append(hr[bound == 1])

                    val_bar.update()
                val_bar.close()

            trues = np.concatenate(trues)
            preds = np.concatenate(preds)
            rmse_val = mean_squared_error(trues, preds, squared=False)

            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results['val_rmse'].append(rmse_val)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'Val_RMSE': results['val_rmse']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + '/' + 'train_logs.csv', index_label='Epoch')

            early_stopping(rmse_val, netG, out_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Updating G_learning rate to {}'.format(schedulerG.get_last_lr()[0]))
            print('Updating D_learning rate to {}'.format(schedulerD.get_last_lr()[0]))

    # test
    if I == 1:
        out_path = 'training_results/' + setting
        netG = Generator(opt)
        if torch.cuda.is_available():
            netG.cuda()
        netG.load_state_dict(torch.load(os.path.join(out_path, "checkpoint.pth")))

        netG.eval()
        lrs = []
        preds = []
        trues = []
        with torch.no_grad():
            for lr, hr, _, _, _ in test_loader:
                lr = lr.float().cuda()
                hr = hr.float().cuda()

                sr = netG(lr)

                sr = sr.detach().cpu().numpy() # b, c, h, w
                hr = hr.detach().cpu().numpy()
                lr = lr.detach().cpu().numpy()

                lrs.append(lr)
                preds.append(sr)
                trues.append(hr)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        lrs = np.concatenate(lrs, axis=0)

        np.savez(os.path.join(out_path, "test_results.npz"), preds=preds, trues=trues, lrs=lrs)
    
    # inference
    if I == 2:
        out_path = 'training_results/' + setting
        netG = Generator(opt)
        if torch.cuda.is_available():
            netG.cuda()
        netG.load_state_dict(torch.load(os.path.join(out_path, "checkpoint.pth")))

        # load data
        models = ["access_cm2", "bcc_csm2_mr", "canesm5_canoe", "cnrm_cm6_1_hr", "miroc6", "miroc_es2l", "mri_esm2_0", 
        "noresm2_mm", "taiesm1", "cesm2", "cmcc_cm2_sr5", "cnrm_esm2_1", "ec_earth3_veg_lr", "kace_1_0_g", "mpi_esm1_2_lr", 
        "noresm2_lm", "ukesm1_0_ll"]
        cmip6_root = "/data/home/scv7343/run/SRGAN-master/data/cmip6"
        exps = ["ssp1_2_6", "ssp2_4_5", "ssp3_7_0", "ssp5_8_5"]
        exp = exps[opt.exp_id]

        amsmqtp_ds = xr.open_dataset("/data/home/scv7343/run/SRGAN-master/data/AMSMQTP_ensemble_monthly_concat.nc")
        lon = amsmqtp_ds.lon.values
        lat = amsmqtp_ds.lat.values
        time_slice = slice("2015-02", "2100")

        if opt.history:
            exp = "historical"
            time_slice = slice("1950", "2014")

        lst = [os.path.join(cmip6_root, f"{x}_{exp}.nc") for x in models]
        data = []
        for x in lst:
            ds = xr.open_dataset(x)
            ds = ds.sel(time=time_slice, lon=lon, lat=lat)
            time = ds.time.values
            data.append(ds.mrsos.values / 100)
        data = np.stack(data, axis=1) # T, C, H, W

        preds = []
        netG.eval()
        with torch.no_grad():
            for lr in iterate_batches(data, opt.batch_size):
                lr = torch.tensor(lr).float().cuda()
                sr = netG(lr)

                sr = sr.detach().cpu().numpy() # b, c, h, w

                preds.append(sr)

        preds = np.concatenate(preds, axis=0)
        save_dir = os.path.join("pred", setting + f"_exp_{exp}")
        os.makedirs(save_dir, exist_ok=True)

        ds = xr.Dataset(
            data_vars=dict(
                soil_moisture=(["time", "layer", "lat", "lon"], preds)
            ), 
            coords=dict(
                time=time, lat=lat, lon=lon, layer=list(range(1, 6))
            )
        )
        ds.to_netcdf(os.path.join(save_dir, "pred.nc"))