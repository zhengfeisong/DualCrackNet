import time
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from evaluate.calculate_metric import calculate_metrics
from dataset import *
from tool import *
import torch
import os
from loss import *
from torch import optim
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from model.dualcracknet import DualCrackNet

open_flag = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Training device, for example cuda:0, cuda:1 or cpu')
    
    parser.add_argument('--data_root', default='RCD900', help='data path')
    parser.add_argument('--savepath', default='weight/', help='Checkpoint root; each run creates a timestamped subdirectory')

    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Legacy argument; unused (Adam weight decay is 0)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_workers', type=int, default=4, help='number of threads in dataloader')
    parser.add_argument('--gamma', type=float, default=0.1, help='Legacy argument; unused (learning rate decays by 0.99 each epoch)')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=5, help='Legacy argument; unused (learning rate updates each epoch)')
    parser.add_argument('--resume_model', type=str, default=None,
                        help='Initialize from model weights; optimizer and epoch counter start fresh')
    parser.add_argument('--print_freq', type=int, default=80, help='the frequency of printing losses (iterations)')
    parser.add_argument('--save_freq', type=int, default=50, help='Save model weights every this many epochs')
    args = parser.parse_args()
    for name in ('batch_size', 'n_epochs', 'print_freq', 'save_freq'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.n_workers < 0 or args.lr <= 0:
        parser.error('workers must be nonnegative; lr must be positive')
    return args

def train(cfg):
    global open_flag
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    print(local_time)
    formatted_time = time.strftime("%Y%m%d_%H%M%S", local_time)
    target_folder = os.path.join(os.path.expanduser(cfg.savepath), formatted_time)

    cfg.device = torch.device(cfg.device)
    if cfg.device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available; use --device cpu')
        device_index = cfg.device.index if cfg.device.index is not None else torch.cuda.current_device()
        if device_index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {device_index} is not available')
        torch.cuda.set_device(device_index)
        cfg.device = torch.device('cuda', device_index)
    net = DualCrackNet().to(cfg.device)
    if cfg.resume_model:
        path = os.path.expanduser(cfg.resume_model)
        try:
            saved = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            saved = torch.load(path, map_location='cpu')
        state = saved['model'] if 'model' in saved else saved
        net.load_state_dict(state, strict=True)
        print(f'Loaded model weights from {path}; optimizer and epoch counter start fresh.')
    os.makedirs(target_folder, exist_ok=False)
    print(f'Checkpoint directory: {os.path.abspath(target_folder)}')
    print("Start time:", formatted_time)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    train_set=CrackDataset(cfg.data_root,'train')
    val_set=CrackDataset(cfg.data_root,'valid')
    F1Score_base=0
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.n_workers, pin_memory=cfg.device.type == 'cuda')
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1, shuffle=True,
        num_workers=cfg.n_workers, pin_memory=cfg.device.type == 'cuda')
    ce_loss = CrossEntropyLoss()
    
    eval = 5
    freeze_epoch = 0
    criterion = BceDiceLoss()
    bbox_criterion = bbox_iou_loss
    
    for epoch in range(cfg.n_epochs):
        # print(open_flag)
        loss_epoch = []
        for param_group in optimizer.param_groups:
            lr=cfg.lr*0.99**(epoch)
            param_group['lr'] = lr
        i=0
        print('lr=%f' % lr)
        net.train()
        
        for batch in train_loader:
            img = batch['image']
            lab = batch['mask']
            bboxes = batch['bboxes']
            lab=lab.type(torch.LongTensor).float()
            img=img.type(torch.FloatTensor)
            img, lab, bboxes = img.to(cfg.device), lab.to(cfg.device), bboxes.to(cfg.device)
            lab=lab.permute(1,0,2,3)
            lab=lab.squeeze(0)
            
            segmentation_output, B, bbox_pred = net(img) 
            batch_size, _, h, w = segmentation_output[0].shape

            if open_flag == 1:
                preds = crop_and_segment(segmentation_output[0], bbox_pred, original_size=(h, w))
            else:
                preds = segmentation_output[0]
                        
            loss1 = criterion(preds,lab)*10
            loss2 = criterion(segmentation_output[1],lab)*10
            loss3 = criterion(segmentation_output[2],lab)*10
            
            loss_initial =  10*loss1+loss2+loss3
            
            bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)      # [8, 4]
            bboxes = bboxes.view(bboxes.size(0), 4)                # [8, 4]
            bbox_loss_value = bbox_criterion(bbox_pred, bboxes) * 15

            boundary = batch['boundary'].to(cfg.device)
            mask_type = torch.float32 if net.num_classes == 1 else torch.long
            boundary = boundary.to(device=cfg.device, dtype=mask_type)
            criteria  = DiceBCELoss()
            loss_boundary = criteria(B.float(), boundary.float()) * 10
            
            loss = 0.85 * loss_initial + 0.15 * loss_boundary + bbox_loss_value * 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())
            i+=1
            if i % cfg.print_freq == 0 and i!=0:
                print('### iteration %5d of loss %5f epoch %5d'%
                      (i + 1,
                        float(np.array(loss_epoch).mean()),epoch))
        if (epoch + 1) % cfg.save_freq == 0:
            torch.save(net.state_dict(), target_folder + f'/model_epoch_{epoch + 1}.pth')
            print(f'Model weights saved at epoch {epoch + 1}')
            
        if (epoch+1)%eval==0:
            conf_mat =np.zeros((2,2),dtype=np.float32)# np.zeros((dataset.dataset.num_labels, dataset.dataset.num_labels), dtype=np.float)
            net.eval()
            with torch.no_grad():
                for batch in val_loader:
                    img = batch['image']
                    lab = batch['mask']
                    bboxes = batch['bboxes']
                    lab=lab.type(torch.LongTensor)
                    img=img.type(torch.FloatTensor)
                    img, lab, bboxes = img.to(cfg.device), lab.to(cfg.device), bboxes.to(cfg.device)
                    pred = net(img) #todo
                    lab=lab.permute(1,0,2,3)
                
                    gt=lab.cpu().numpy()
                    pred = pred[0]
                    pred = pred.data.cpu().numpy()
    
                    pred = (pred >= 0.5).astype(np.int32) 

                    oriSize=(320,320)
                
                    gt=gt.squeeze(0)
                    # gt = np.expand_dims(cv2.resize(np.squeeze(gt, axis=0), oriSize, interpolation=cv2.INTER_NEAREST), axis=0)
                    pred = pred.squeeze(0)
                    conf_mat += confusion_matrix(gt, pred, 2)
                    i+=1
            _, _, _, F_score, _, _ = getScores(conf_mat)
            F_score = F_score.real
            print(F_score)
            if F_score > 0.85 and open_flag == 0:
                open_flag = 1
            if F_score > F1Score_base:
                print(F1Score_base)
                F1Score_base=F_score
                formatted_f1score = "{:.5f}".format(F_score)
                torch.save(net.state_dict(), target_folder+'/best_model'+'.pth')
                print('saving the model at the end of epoch %d' % (epoch))

    timestamp = time.time()
    local_time = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    print("End time:", formatted_time)

if __name__ == "__main__":
    cfg=parse_args()
    train(cfg)
    

            
