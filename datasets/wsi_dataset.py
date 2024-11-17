import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.transforms import Polar
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import glob
import random
from tqdm import trange


class PairData(Data):
    def __init__(self, x=None, edge_index=None):
        super().__init__(x=x, edge_index=edge_index)

    def __inc__(self, key, value, *args, **kwargs):
        if not hasattr(Data, key):
            if 'edge' in key:
                return getattr(self, 'x').size(0)
            else:
                return 0
        else:
            return super(PairData, self).__inc__(key, value)


def data_transform(data):
    if data is None:
        return None
    x = data.x
    edge = data.edge_index

    n = len(x)
    rate = random.uniform(0.1, 0.3)
    rows_to_noise = np.random.choice(n, int(n * rate), replace=False)
    noise = torch.randn_like(x[rows_to_noise]) * 0.1
    x[rows_to_noise] += noise

    rate = random.uniform(0.1, 0.3)
    rows_to_drop = np.random.choice(n, int(n * rate), replace=False)
    x[rows_to_drop] = 0

    # drop edges
    rate = random.uniform(0.1, 0.3)
    m = edge.shape[1]
    k = m - int(rate * m)
    columns = np.random.choice(m, k, replace=False)
    edge = edge[:, columns]

    if hasattr(data, 'edge_latent'):
        edge_latent = data.edge_latent
        rate = random.uniform(0.1, 0.3)
        m = edge_latent.shape[0]
        k = m - int(rate * m)
        columns = np.random.choice(m, k, replace=False)
        edge_latent = edge_latent[columns, :]
        data.edge_latent = edge_latent

    data.x = x
    data.edge_index = edge

    return data

class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False, seed=7, print_info=True,
                 patient_strat=False, label_col='survival'):
        super(Generic_WSI_Survival_Dataset, self).__init__()
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None

        try:
            slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False, encoding='gbk')
        except:
            slide_data = pd.read_csv(csv_path, index_col=0, low_memory=False)

        assert label_col in slide_data.columns
        self.label_col = label_col
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        patients_df = slide_data.drop_duplicates(['case_id']).copy()

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)

        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values}

        # make order ['survival_days', 'disc_label', 'case_id', 'slide_id', 'label', 'censorship']
        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        slide_data['case_id'] = slide_data['case_id'].astype(str)

        self.slide_data = slide_data

        self.filelist = list(self.slide_data['slide_id'])

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.slide_data)

    def get_split_from_df(self, all_splits, split_key: str = 'train', use_transform=False):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        try:
            split = split.astype(int).astype(str)
        except:
            split = split.astype(str)

        self.slide_data['case_id'] = self.slide_data['case_id'].astype(str)
        if len(split) == 0:
            return None
        mask = self.slide_data['case_id'].isin(split.tolist())

        df_slice = self.slide_data[mask].reset_index(drop=True)

        split = Generic_Split(df_slice, data_dir=self.data_dir, label_col=self.label_col, mode=split_key,
                              use_transform=use_transform, in_memory=self.in_memory, bar=self.bar)
        return split

    def return_splits(self, csv_path=None, split_key='train_val', use_transform=True):
        assert csv_path
        all_splits = pd.read_csv(csv_path)
        # print(all_splits)
        if '_' in split_key:
            ans = []
            if 'train' in split_key:
                print('make train split')
                train_split = self.get_split_from_df(all_splits=all_splits, split_key='train',
                                                     use_transform=use_transform)
                ans.append(train_split)
            if 'val' in split_key:
                print('make val split')
                val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
                ans.append(val_split)
            if 'test' in split_key:
                print('make test split')
                test_split = self.get_split_from_df(all_splits=all_splits, split_key='test')
                ans.append(test_split)

            if len(ans) == 1:
                return ans[0]
            return ans

        else:
            print(f'make {split_key} split')
            split = self.get_split_from_df(all_splits=all_splits, split_key=split_key)
            return split

    def get(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, args_o, in_memory=False, bar=False, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]
        self.in_memory = in_memory
        self.bar = bar
        global args
        args = args_o


class Generic_Split(Dataset):
    def __init__(self, slide_data, data_dir=None, label_col=None,
                 mode='train', use_transform=True,
                 in_memory=False, bar=False
                 ):
        super().__init__()
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.label_col = label_col

        self.mode = mode
        self.use_transform = use_transform

        self.in_memory = in_memory

        self.censorship = list(self.slide_data['censorship'])

        self.data_in_memory = []
        if self.in_memory:
            a = time.time()
            print('Load data to memory', flush=True)
            print(len(self.slide_data))
            if bar:
                rg = trange(len(self.slide_data))
            else:
                rg = range(len(self.slide_data))
            self.censorship = []
            self.dataset = []
            for i in rg:
                tmp = self.get_data_from_idx(i)
                if tmp is not None:
                    self.data_in_memory.append(tmp)
                    self.censorship.append(self.slide_data['censorship'][i])
            print(f"Load data finish, cost {time.time() - a}s", flush=True)
        else:
            print('Load data from paths', flush=True)

    def get_data_from_idx(self, idx):
        case_id = self.slide_data['case_id'][idx]
        try:
            event_time = int(self.slide_data[self.label_col][idx])
        except:
            print('event_time error')
            return None
        c = self.slide_data['censorship'][idx]
        slide_ids = self.slide_data['slide_id'][idx]

        wsi_path = ''
        for pth in self.data_dir:
            tmp_pth = f"{pth}/{slide_ids.rstrip('.svs').rstrip('.ndpi')}.pt"
            if os.path.exists(tmp_pth):
                wsi_path = tmp_pth
                break
        if wsi_path == '':
            return None

        data_origin = torch.load(wsi_path)

        data_t = PairData(data_origin.x, data_origin.edge_index)

        if hasattr(data_origin, 'edge_latent'):
            data_t.edge_latent = data_origin.edge_latent.transpose(0, 1)

        data_t.centroid = data_origin.centroid
        data_t.event_time = event_time
        data_t.c = c
        data_t.slide_id = slide_ids.split('/')[-1].rstrip('.svs').rstrip('.ndpi')

        if args.use_pre_proto:
            tmp_cls = torch.zeros(data_t.x.shape[0]) - 1
            data_t.patch_classify_type = tmp_cls
        else:
            try:
                if hasattr(args, 'cls_attr'):
                    data_t.patch_classify_type = getattr(data_origin, args.cls_attr)
                else:
                    data_t.patch_classify_type = data_origin.patch_classify_type
            except Exception as e:
                tmp_cls = torch.zeros(data_t.x.shape[0]) + args.nr_types - 1
                data_t.patch_classify_type = tmp_cls

        data_t.case_id = case_id

        return data_t

    def get(self, idx):
        if self.in_memory:
            data_t = copy.deepcopy(self.data_in_memory[idx])
        else:
            data_t = self.get_data_from_idx(idx)
        if self.mode == 'train' and self.use_transform:
            data_t = data_transform(data_t)
        return data_t

    def len(self):
        if len(self.data_in_memory) > 0:
            return len(self.data_in_memory)
        else:
            return len(self.slide_data)
