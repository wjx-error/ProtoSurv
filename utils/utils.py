import pickle

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import pdb
import random
import os

import torch

from torch.utils.data.sampler import Sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch
import yaml
from argparse import Namespace

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ordered_yaml():
    """
    yaml orderedDict support
    """
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_custom_exp_code(args):
    # param_code 模型参数 包括学习率 alpha
    # exp_code 数据集参数 包括使用的split，边类型
    param_code = args.model_type
    if args.resample > 0:
        param_code += '_resample'
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv)
    if args.gc != 1:
        param_code += '_gc%s' % str(args.gc)

    param_code += f'_s{args.seed}'

    args.param_code = param_code
    args.dataset_path = 'dataset_csv'

    exp_code = args.project_name
    # exp_code = ''
    exp_code += '_%s' % args.split_dir
    if isinstance(args.data_dir, list):
        for x in args.data_dir:
            exp_code += '_%s' % x.split('/')[-1]
    else:
        exp_code += '_%s' % args.data_dir.split('/')[-1]

    args.exp_code = exp_code

    return args


import time


class Sampler_custom(Sampler):
    def __init__(self, censorship, batch_size):
        event_list = np.where(np.array(censorship) == 0)[0]  # 0 event(died)
        censor_list = np.where(np.array(censorship) == 1)[0]  # 1 survival

        print('init Sampler_custom')
        print('event_list', len(event_list))
        print('censor_list', len(censor_list))

        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

        self.min_event = 2
        if self.batch_size < 4:
            self.min_event = 1

        self.size = (len(event_list) + len(censor_list)) // batch_size

    def __iter__(self):
        # random.seed(int(time.time()))
        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Event_idx = list(Event_idx)
        Censored_idx = list(Censored_idx)
        while 1:
            if len(Event_idx) == 0 or (len(Event_idx) + len(Censored_idx)) < self.batch_size:
                break
            event_rate = random.uniform(0.2, 1)
            event_num = max(self.min_event, int(event_rate * self.batch_size))
            if len(Event_idx) < event_num:
                event_num = len(Event_idx)

            censor_num = self.batch_size - event_num
            if len(Censored_idx) < censor_num:
                censor_num = len(Censored_idx)
                event_num = self.batch_size - censor_num

            event_batch_select = random.sample(Event_idx, event_num)
            censor_batch_select = random.sample(Censored_idx, censor_num)

            Event_idx = list(set(Event_idx) - set(event_batch_select))
            Censored_idx = list(set(Censored_idx) - set(censor_batch_select))

            selected_list = event_batch_select + censor_batch_select
            random.shuffle(selected_list)
            train_batch_sampler.append(selected_list)

        random.shuffle(train_batch_sampler)
        self.size = len(train_batch_sampler)
        # print('run custom resample',len(train_batch_sampler))

        return iter(train_batch_sampler)

    def __len__(self):
        return int(self.size)


def get_args(args, configs_dir='configs', added_dict=None):
    project_nm = args.project_name
    print('project_name', project_nm)

    if configs_dir not in args.yml_opt_path:
        args.yml_opt_path = f"{configs_dir}/{args.yml_opt_path}"
    if '.yml' not in args.yml_opt_path:
        args.yml_opt_path = f"{args.yml_opt_path}.yml"

    print('args.yml_opt_path', args.yml_opt_path, flush=True)

    with open(args.yml_opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {args.yml_opt_path}")
    args = Namespace(**config)

    if added_dict is not None:
        for key, value in added_dict.items():
            print(f'added_dict set args {key} to {value}', flush=True)
            setattr(args, key, value)

    args.project_name = project_nm
    print('Current project:', args.project_name)
    args.reg = float(args.reg)
    args.task_type = 'survival'

    args.csv_path = args.csv_path.replace('(project_name)', args.project_name)
    args.split_dir = args.split_dir.replace('(project_name)', args.project_name)
    args.data_dir = args.data_dir.replace('(project_name)', args.project_name)

    if not hasattr(args, 'use_pre_proto'):
        args.use_pre_proto = False
    args.use_pre_proto = bool(args.use_pre_proto)
    if not hasattr(args, 'fp16_precision'):
        args.fp16_precision = False
    args.fp16_precision = bool(args.fp16_precision)
    if not hasattr(args, 'in_memory'):
        args.in_memory = False

    if args.use_pre_proto:
        args.comp_alpha = 0
    args.pin_memory = True
    if args.in_memory:
        args.num_workers = 0
        args.pin_memory = False

    ### Creates Experiment Code from argparse + Folder Name to Save Results
    args = get_custom_exp_code(args)
    args.task = '_'.join(args.split_dir.split('_')[:2])
    print("Experiment Name:", args.exp_code)
    args.results_dir = f"{args.results_dir}/{args.which_splits}/{args.model_type}/{str(args.param_code)}/{str(args.exp_code)}/"

    return args


from models.model_protosurv_v1 import LINKX_PROTO_oldv


def get_model(args):
    print('\nInit Model...')
    if args.model_type == 'protosurv_v1':
        model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg,
                      'n_classes': args.n_classes, 'input_dim': args.input_dim,
                      'nr_types': args.nr_types, 'num_proto': args.num_proto,
                      'pre_proto_pth': f"./pre_prototypes/{args.project_name}.pt",
                      'pre_proto': args.use_pre_proto
                      }
        model = LINKX_PROTO_oldv(**model_dict)
    else:
        raise NotImplementedError
    print('Done')
    return model
