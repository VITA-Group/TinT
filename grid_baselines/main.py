# from torch.utils.data import Dataset,BatchSampler,SequentialSampler,RandomSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, sys
import numpy as np
import math, random
import configargparse
from tqdm import tqdm, trange

from utils import get_args

from models import TrafficUNet2D, TrafficUNet3D, TrafficUNet2D_LSTM, \
    TrafficViT, TrafficSwin, TrafficViViT
from data import IteratableTrafficDataset as TrafficDataset

def masked_mape(gt, pred):
    # gt = gt.reshape(gt.shape[0], gt.shape[1], -1)
    # pred = pred.reshape(pred.shape[0], pred.shape[0], -1)
    mask = (torch.abs(pred) == 0)
    mape = torch.abs((pred - gt) / pred)
    mape[mask] = 0.0
    # mape = torch.abs((pred - gt)[mask] / pred[mask])
    mape = mape.sum(-1).sum(0) / (mape.shape[-1] * mape.shape[0])
    return mape

def rmse(gt, pred):
    # gt = gt.reshape(gt.shape[0], -1)
    # pred = pred.reshape(pred.shape[0], -1)
    mse = ((pred - gt) ** 2).sum(-1).sum(0) / (pred.shape[-1] * pred.shape[0])
    return torch.sqrt(mse)

def mae(gt, pred):
    mae = torch.abs(pred - gt).sum(-1).sum(0) / (pred.shape[-1] * pred.shape[0])
    return mae

@torch.no_grad()
def eval(model, eval_loader, args, reduce=True):

    total_mae = 0.
    total_mape = 0.
    total_mse = 0.
    total_count = 0

    all_true = []
    all_preds = []

    for i, (x, y_true) in enumerate(tqdm(eval_loader)):
        x, y_true = x.float().to(args.device), y_true.float().to(args.device)
        y_pred = model(x)

        all_true.append(y_true.cpu())
        all_preds.append(y_pred.cpu())

    y_true = torch.cat(all_true, 0)
    y_pred = torch.cat(all_preds, 0)
    if reduce:
        y_true = y_true.reshape(y_true.shape[0], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)
        return mae(y_true, y_pred).item(), masked_mape(y_true, y_pred).item(), rmse(y_true, y_pred).item()
    else:
        y_true = y_true.reshape(y_true.shape[0], y_true.shape[1], -1)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1)

        return mae(y_true, y_pred), masked_mape(y_true, y_pred), rmse(y_true, y_pred)

    # if reduce:
    #     total_mae += mae(y_true, y_pred).item()
    #     total_mape += masked_mape(y_true, y_pred).item()
    #     total_mse += rmse(y_true, y_pred).item()
    # else:
    #     y_pred = y_pred.permute([0, 2, 3, 4 ,1]).reshape(-1, gt.shape[-1])
    #     y_pred = y_pred.permute([0, 2, 3, 4 ,1]).reshape(-1, gt.shape[-1])
    #     total_mae = total_mae + torch.abs(y_pred - y_true).sum(0)
    #     total_mape = total_mape + masked_mape(y_true, y_pred, reduce=False)
    #     total_mse = total_mse + rmse(y_true, y_pred, reduce=False)

        # total_count += y_pred.reshape(-1, y_pred.shape[-1]).shape[0]
    #     total_count += y_pred.reshape(-1).shape[0]

    # if reduce:
    #     return total_mae / total_count, total_mape / total_count, math.sqrt(total_mse / total_count)
    # else:
    #     return total_mae / total_count, total_mape / total_count, torch.sqrt(total_mse / total_count)

def train(args):
    train_dataset = TrafficDataset(args.data_dir, args.city, args.region, 'train', shuffle=True,
        crop_size=(args.crop_size, args.crop_size), t_len=args.pred_len, t_stride=args.t_subsample)
    valid_dataset = TrafficDataset(args.data_dir, args.city, args.region, 'valid', shuffle=False,
        crop_size=(args.crop_size, args.crop_size), t_len=args.pred_len, t_stride=args.t_subsample)
    test_dataset = TrafficDataset(args.data_dir, args.city, args.region, 'test', shuffle=False,
        crop_size=(args.crop_size, args.crop_size), t_len=args.pred_len, t_stride=args.t_subsample)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=args.pin_mem)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=args.pin_mem)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=args.pin_mem)


    print('Building model:', args.model)
    if args.model == 'UNet':
        model = TrafficUNet2D(args).to(args.device)
    if args.model == 'UNetLSTM':
        model = TrafficUNet2D_LSTM(args).to(args.device)
    elif args.model == 'UNet3D':
        model = TrafficUNet3D(args).to(args.device)
    elif args.model == 'ViT':
        model = TrafficViT(args).to(args.device)
    elif args.model == 'Swin':
        model = TrafficSwin(args).to(args.device)
    elif args.model == 'ViViT':
        model = TrafficViViT(args).to(args.device)
    else:
        raise ValueError('Unknown model name', args.model)

    loss_fn = nn.MSELoss(reduction='mean')
    opt_funcs = {'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}
    optimizer = opt_funcs[args.opt](model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    start_epoch = 0

    if os.path.exists(args.resume):
        ckpt_dict = torch.load(args.resume)
        start_epoch = ckpt_dict['epoch']
        model.load_state_dict(ckpt_dict['model'])
        optimizer.load_state_dict(ckpt_dict['optimizer'])

    if args.eval:
        test_mae, test_mape, test_rmse = eval(model, test_loader, args)
        print(f'[TEST] MAE: {test_mae:.04f} MAPE: {test_mape:.04f} RMSE: {test_rmse:.04f}')
        # print(f'[TEST] {test_mae:.04f} & {test_mape:.04f} & {test_rmse:.04f}')
        # test_mae, test_mape, test_rmse = eval(model, test_loader, args, reduce=False)
        # print(test_mae.shape, test_mape.shape, test_rmse.shape)
        # np.savez(f'{args.city.lower()}_{args.region.lower()}_{args.model.lower()}.npz', mae=test_mae.numpy(), mape=test_mape.numpy(), rmse=test_rmse.numpy())
        return

    for epoch in range(start_epoch, args.epochs):
        for i, (x, y_true) in enumerate(tqdm(train_loader)):
            x, y_true = x.float().to(args.device), y_true.float().to(args.device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_mae, test_mape, test_rmse = eval(model, test_loader, args)
        print(f'Epoch: {epoch} Train loss: {loss:.4f} Test MAE: {test_mae:.4f}')

        with open(os.path.join(args.log_dir, 'log.txt'), 'a') as f:
            print(f'Epoch: {epoch} Train loss: {loss} Test MAE: {test_mae}', file=f)

        if epoch % 10 == 0:
            ckpt_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(ckpt_dict, os.path.join(args.log_dir, 'ckpts', f'{epoch:04d}.ckpt'))

    return


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, required=True, help='config file path')
    parser.add_argument("--log_dir", type=str, required=True, help='path to store ckpts and logs')
    parser.add_argument("--resume", type=str, default='', help='path to reload checkpoint')
    parser.add_argument("--gpuid", type=int, default=0, help='cuda number')
    parser.add_argument("--eval", action='store_true', default=False, help='evaluation only')

    parser.add_argument("--data_dir", type=str, required=True, help='path to the data')
    parser.add_argument("--city", type=str, default='BERLIN', choices=['BERLIN', 'ISTANBUL', 'BANGKOK'],
                        help='city name of the dataset')
    parser.add_argument("--region", type=str, default='DT', choices=['DT', 'Rural'],
                        help='region of city to crop the dataset')
    parser.add_argument("--t_subsample", type=int, default=1, help='subsampling rate of time axis for dataset')
    parser.add_argument("--crop_size", type=int, default=32, help='crop size of the selected region')
    parser.add_argument("--num_workers", type=int, default=8, help='number of data loading workers')
    parser.add_argument("--pin_mem", action='store_true', default=False, help='turn on pin memory for data loading')
    parser.add_argument("--no_pin_mem", action='store_false', dest='pin_mem', help='turn off pin memory for data loading')
    parser.set_defaults(pin_mem=True)


    parser.add_argument("--model", "-m", type=str, default='UNet3D', help='model name',
                        choices=['UNet', 'UNetLSTM', 'UNet3D', 'ViT', 'Swin', 'ViViT'])
    parser.add_argument("--pred_len", type=int, default=12, help='forecast time period')
    parser.add_argument("--opt", type=str, default='SGD',
                        choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument("--lr", type=float, default=0.005, help='learning rate')
    parser.add_argument("--wdecay", type=float, default=0.0, help='weight decay')
    parser.add_argument("--epochs", type=int, default=100, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size')
    parser.add_argument("--dim_feature", type=int, default=8, help='dimension of input feature')
    parser.add_argument("--dim_hidden", type=int, default=64, help='dimension of hidden feature')

    # options for ViT
    parser.add_argument("--num_layers", type=int, default=8, help='number of layers (only for vits)')
    parser.add_argument("--num_heads", type=int, default=4, help='number of heads for attention')
    parser.add_argument("--patch_size", type=int, default=16, help='patch size for vit tokenization')
    parser.add_argument("--window_size", type=int, default=7, help='window size for swin transformer')

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', args.device)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.join(args.log_dir, 'ckpts'), exist_ok=True)
    print('Logging into directory:', args.log_dir)

    train(args)
