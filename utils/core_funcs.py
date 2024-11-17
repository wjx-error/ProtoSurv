import copy
from argparse import Namespace
from collections import OrderedDict
import os
import pickle

from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
import torch
from datasets.dataset_generic import save_splits
from utils.utils import *
from utils.utils_loss import *
from torch.cuda.amp import GradScaler, autocast
import gc
import tqdm

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping_cindex:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_cindex = None
        self.best_cindex_num = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, cindex, model, ckpt_name='best_checkpoint.pt'):
        cindex_num = cindex[0]
        if epoch < self.warmup:
            pass
        elif self.best_cindex == None:
            self.best_cindex = cindex
            self.best_cindex_num = cindex_num
            self.save_checkpoint(cindex, cindex_num, model, ckpt_name)

        elif cindex_num <= self.best_cindex_num:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best val cindex:{self.best_cindex}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.save_checkpoint(cindex, cindex_num, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_cindex, cindex_num, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation c-index insreased ({self.best_cindex[0]:.4f}, {self.best_cindex[1]:.4f} --> {val_cindex[0]:.4f}, {val_cindex[1]:.4f}).  Saving model to {ckpt_name}...')
            # print(f'Validation c-index insreased ({self.best_cindex:.6f} --> {val_cindex:.6f}).  Saving model to {ckpt_name}...')
        torch.save(model.state_dict(), ckpt_name)
        self.best_cindex = val_cindex
        self.best_cindex_num = cindex_num


def train_loop_survival(epoch, model, loader, optimizer, n_classes, args, writer=None,
                        loss_fn=None, reg_fn=None, lambda_reg=0., gc=16,
                        max_node_num=20_0000, model_type='proto',
                        bar=True):
    if hasattr(model, 'reset_drop'):
        model.reset_drop(train_mode=True)

    scaler = GradScaler(enabled=args.fp16_precision)

    model.train()
    train_loss_surv, train_loss = 0., 0.

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    cnt = 0
    if bar == True:
        loader = tqdm.tqdm(loader)

    # print('autocast', args.fp16_precision)
    cnt = 0
    for batch_idx, data_WSI in enumerate(loader):
        cnt += 1
        data_WSI = [x for x in data_WSI if x is not None]
        if len(data_WSI) == 0:
            continue

        # print('data_WSI', data_WSI)
        event_time = torch.tensor([data.event_time for data in data_WSI])
        c = torch.tensor([data.c for data in data_WSI])

        if torch.sum(c == 0) == 0:
            print('all censored')
            continue

        c = c.to(device)
        event_time = event_time.to(device)
        with autocast(enabled=args.fp16_precision):
            hazards, S, Y_hat, _, others = model(data_WSI)
            loss_dict = {'hazards': hazards, 'S': S, 'c': c, 'event_time': event_time}
            loss = loss_fn(**loss_dict)

            if 'proto' in args.model_type and others is not None and (args.comp_alpha != 0 or args.ortho_beta != 0):
                prototypes_comp = others
                if args.comp_alpha != 0:
                    comploss = get_compatibility_loss_batch(data_WSI, prototypes_comp, args.nr_types)
                    loss += args.comp_alpha * comploss
                if args.ortho_beta != 0:
                    ortholoss = get_orthogonal_regularization_loss(prototypes_comp)
                    loss += args.ortho_beta * ortholoss

            loss_value = loss.item()

        if torch.any(torch.isnan(S)):
            print('error: out nan', flush=True)

        if not torch.isnan(loss):
            loss = loss / gc
            # loss.backward()
            scaler.scale(loss).backward()
        else:
            print('loss nan', flush=True)

        if S.shape[1] > 1:
            risk = -torch.sum(S, dim=1)
        else:
            risk = S

        risk = risk.detach().cpu().numpy()
        train_loss_surv += loss.item()
        train_loss += loss_value

        all_risk_scores.extend(risk.flatten())
        all_censorships.extend(c.cpu().numpy().flatten())
        all_event_times.extend(event_time.cpu().numpy().flatten())

        if (batch_idx + 1) % gc == 0:
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    # print(f'loss: {loss.item():.4f}, ortholoss: {ortholoss.item():.4f}, comploss: {comploss.item():.4f}')

    optimizer.step()
    optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(all_risk_scores)
    train_loss /= len(all_risk_scores)

    all_risk_scores = np.asarray(all_risk_scores).flatten()
    all_censorships = np.asarray(all_censorships).flatten()
    all_event_times = np.asarray(all_event_times).flatten()

    c_index_all = 1 - concordance_index(all_event_times, all_risk_scores)
    c_index = 1 - concordance_index(all_event_times, all_risk_scores, event_observed=(1 - all_censorships))

    print(
        'Epoch: {}, train_loss: {:.4f}, train_c_index_censored: {:.4f},  train_c_index_all: {:.4f}'
            .format(epoch, train_loss, c_index, c_index_all), flush=True)

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)

    return c_index


def validate_survival(epoch, model, loader, n_classes, early_stopping=None,
                      writer=None, loss_fn=None, reg_fn=None, lambda_reg=0.,
                      results_dir=None, cur=None, max_node_num=20_0000,
                      model_type='proto', bar=True, mode='Val'):
    if hasattr(model, 'reset_drop'):
        model.reset_drop(train_mode=False)
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    cnt = 0
    if bar == True:
        loader = tqdm.tqdm(loader)
    for batch_idx, data_WSI in enumerate(loader):
        cnt += 1
        data_WSI = [x for x in data_WSI if x is not None]
        if len(data_WSI) == 0:
            continue
        event_time = torch.tensor([data.event_time for data in data_WSI])

        c = torch.tensor([data.c for data in data_WSI])
        c = c.to(device)
        event_time = event_time.to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(data_WSI)  # return hazards, S, Y_hat, A_raw, results_dict
            # hazards, S, Y_hat, _, _ = model(edge_index=edge_index, node_types=node_types, x=x)

            if torch.any(torch.isnan(S)):
                # print(data_WSI)
                print('error: val out nan', flush=True)

            if S.shape[1] > 1:
                risk = -torch.sum(S, dim=1).cpu().numpy()
            else:
                risk = S.cpu().numpy()

            all_risk_scores.extend(risk)
            all_censorships.extend(c.cpu().numpy())
            all_event_times.extend(event_time.cpu().numpy())

    all_risk_scores = np.asarray(all_risk_scores).flatten()
    all_censorships = np.asarray(all_censorships).flatten()
    all_event_times = np.asarray(all_event_times).flatten()

    # c_index_lifelines = concordance_index(all_event_times, all_risk_scores, event_observed=1 - all_censorships)
    c_index_all = 1 - concordance_index(all_event_times, all_risk_scores)
    c_index = 1 - concordance_index(all_event_times, all_risk_scores, event_observed=(1 - all_censorships))

    if writer:
        writer.add_scalar('val/c-index', c_index, epoch)

    print(
        '{} Epoch: {}, val_c_index_censored: {:.4f}, val_c_index_all: {:.4f}'.format(
            mode, epoch, c_index, c_index_all), flush=True)
    # print('c_index_lifelines',c_index_lifelines)
    # print('val data num',len(all_risk_scores))

    if early_stopping:
        assert results_dir
        all_cindex = [c_index, c_index_all]
        early_stopping(epoch, all_cindex, model,
                       ckpt_name=os.path.join(results_dir, f"s_{cur}_maxcindex_checkpoint.pt"))

        if early_stopping.early_stop:
            print("Early stopping", flush=True)
            return True
    return False


def summary_survival(model, loader, bar=True, hotmap=False, hotmap_pth=None, args=None):
    if hasattr(model, 'reset_drop'):
        model.reset_drop(train_mode=False)
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    patient_results = {}

    cnt = 0
    if bar == True:
        loader = tqdm.tqdm(loader)
    for batch_idx, data_WSI in enumerate(loader):
        cnt += 1
        data_WSI = [x for x in data_WSI if x is not None]
        if len(data_WSI) == 0:
            continue
        event_time = torch.tensor([data.event_time for data in data_WSI])
        c = torch.tensor([data.c for data in data_WSI])
        c = c.to(device)
        event_time = event_time.to(device)

        with torch.no_grad():
            _, S, _, att, _ = model(data_WSI)

        # hotmap
        if hotmap and att is not None:
            att = att[0]  # b,CK,h,d
            att = np.squeeze(att)
            # att = np.mean(att, axis=1)
            att = att[:, 0, :]
            num_proto = args.num_proto
            nr_types = args.nr_types
            cnt = 0
            for i in range(nr_types):
                for j in range(num_proto):
                    tmp_att = att[cnt]
                    draw_hotmap(data_WSI[0], tmp_att, hotmap_pth, back_name=f'{i}_{j}')
                    cnt += 1

        if S.shape[1] > 1:
            risk = -torch.sum(S, dim=1).cpu().numpy()
        else:
            risk = S.cpu().numpy()

        all_risk_scores.extend(risk)
        all_censorships.extend(c.cpu().numpy())
        all_event_times.extend(event_time.cpu().numpy())

        for idx, x in enumerate(data_WSI):
            slide_id = x.slide_id
            risk_t = risk[idx]
            # print('risk_t',risk_t)
            # print(type(risk_t.cpu().numpy()))
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': float(risk_t),
                                               'survival': x.event_time, 'censorship': x.c}})

    all_risk_scores = np.asarray(all_risk_scores).flatten()
    all_censorships = np.asarray(all_censorships).flatten()
    all_event_times = np.asarray(all_event_times).flatten()

    # print(len(all_risk_scores))
    # print('<= 36')
    # all_risk_scores = all_risk_scores[all_event_times <= 36]
    # all_censorships = all_censorships[all_event_times <= 36]
    # all_event_times = all_event_times[all_event_times <= 36]
    # print(len(all_risk_scores))

    c_index_all = 1 - concordance_index(all_event_times, all_risk_scores)
    c_index = 1 - concordance_index(all_event_times, all_risk_scores, event_observed=(1 - all_censorships))
    return patient_results, c_index, c_index_all, cnt


def draw_hotmap(data, att, hotmap_pth, back_name=''):
    coord = data.centroid
    coord = np.asarray(coord, dtype=np.int32)

    # att = att.cpu().numpy()
    att = np.log(att)
    att = 255 * (att - np.min(att)) / (np.max(att) - np.min(att))

    b, a = np.max(coord[:, 0]), np.max(coord[:, 1])
    n = coord.shape[0]

    mask = np.zeros([int(a * 1.05), int(b * 1.05)])
    for i in range(n):
        mask[coord[i, 1], coord[i, 0]] = att[i]

    color_map_mood = cv2.COLORMAP_JET

    mask = mask.astype(np.uint8)
    colored_image = cv2.applyColorMap(mask, color_map_mood)

    ### colorbar
    # colorbar_width = 8
    # colorbar_height = int(mask.shape[0] * 0.9)
    # colorbar = np.linspace(255, 0, colorbar_height).astype(np.uint8)
    # colorbar = cv2.applyColorMap(colorbar, color_map_mood)
    # colorbar = cv2.resize(colorbar, (colorbar_width, colorbar_height))
    #
    # desired_height, desired_width = mask.shape[0], 20
    # padding_height_before = (desired_height - colorbar.shape[0]) // 2
    # padding_height_after = desired_height - colorbar.shape[0] - padding_height_before
    # padding_width_before = (desired_width - colorbar.shape[1]) // 2
    # # padding_width_after = desired_width - colorbar.shape[1] - padding_width_before
    # padding_width = [(padding_height_before, padding_height_after),
    #                  (5, 5),
    #                  (0, 0)]
    #
    # colorbar = np.pad(colorbar, padding_width, mode='constant', constant_values=0)
    # colored_image = np.hstack((colored_image, colorbar))

    save_name = data.slide_id
    # hotmap_pth='./hotmap'

    # print(hotmap_pth)
    # split_lst = hotmap_pth.split('_')
    # print(split_lst)
    # split_lst = [x for x in split_lst if 'time' in x]
    # split_name = split_lst[0]

    os.makedirs(f'{hotmap_pth}/{save_name}', exist_ok=True)
    cv2.imwrite(f'{hotmap_pth}/{save_name}/{back_name}.png', colored_image)
