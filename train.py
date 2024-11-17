import gc
import os
import argparse
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code
from datasets.wsi_dataset import Generic_MIL_Survival_Dataset

### PyTorch Imports
import torch
from utils.utils import ordered_yaml, seed_torch, get_args
import copy


def main(args):
    #### Create Results Directory
    os.makedirs(args.results_dir, exist_ok=True)
    latest_val_cindex = []
    latest_val_cindex_all = []
    latest_test_cindex = []
    latest_test_cindex_all = []

    split_list = os.listdir(args.split_dir)
    split_list = [x for x in split_list if '.csv' in x]
    split_list.sort()
    print('split_list', split_list)

    for i, csv_nm in enumerate(split_list):
        print(f"Start Fold {i}, csv: {csv_nm}")
        start = timer()
        seed_torch(args.seed)

        ### Gets the Train + Val Dataset Loader.
        if args.testing:
            train_dataset, val_dataset, test_dataset = dataset.return_splits(
                csv_path=f'{args.split_dir}/{csv_nm}', split_key='train_val_test')
            print('training: {}, validation: {},  Test: {}'
                  .format(len(train_dataset), len(val_dataset), len(test_dataset)))
            datasets = (train_dataset, val_dataset, test_dataset)
        else:
            train_dataset, val_dataset = dataset.return_splits(csv_path=f'{args.split_dir}/{csv_nm}',
                                                               split_key='train_val')
            print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
            datasets = (train_dataset, val_dataset, None)

        if args.testing:
            val_result, test_result = train(datasets, args, cur=i, bar=args.bar)
            val_latest, cindex_val_latest, val_c_index_all_latest = val_result
            test_latest, cindex_test_latest, test_c_index_all_latest = test_result

            latest_val_cindex.append(cindex_val_latest)
            latest_val_cindex_all.append(val_c_index_all_latest)
            save_pkl(f"{args.results_dir}/{csv_nm}_val_results.pkl", val_latest)
            latest_test_cindex.append(cindex_test_latest)
            latest_test_cindex_all.append(test_c_index_all_latest)
            save_pkl(f"{args.results_dir}/{csv_nm}_test_results.pkl", test_latest)

        else:
            val_latest, cindex_val_latest, val_c_index_all_latest = train(datasets, args, cur=i, bar=args.bar)
            latest_val_cindex.append(cindex_val_latest)
            latest_val_cindex_all.append(val_c_index_all_latest)
            save_pkl(f"{args.results_dir}/{csv_nm}_val_results.pkl", val_latest)

        end = timer()
        print(f'Fold {i} Time: {(end - start):.2f} seconds')
        print()
        gc.collect()

    folds = copy.deepcopy(split_list)
    folds = [x.rstrip('.csv') for x in folds]
    folds.append('average')
    latest_val_cindex.append(np.mean(latest_val_cindex))
    latest_val_cindex_all.append(np.mean(latest_val_cindex_all))
    folds.append('std')
    latest_val_cindex.append(np.std(latest_val_cindex))
    latest_val_cindex_all.append(np.std(latest_val_cindex_all))

    if args.testing:
        latest_test_cindex.append(np.mean(latest_test_cindex))
        latest_test_cindex_all.append(np.mean(latest_test_cindex_all))

        latest_test_cindex.append(np.std(latest_test_cindex))
        latest_test_cindex_all.append(np.std(latest_test_cindex_all))

        results_latest_df = pd.DataFrame(
            {'folds': folds, 'val_cindex': latest_val_cindex, 'val_cindex_all': latest_val_cindex_all,
             'test_cindex': latest_test_cindex, 'test_cindex_all': latest_test_cindex_all})
    else:
        results_latest_df = pd.DataFrame(
            {'folds': folds, 'val_cindex': latest_val_cindex, 'val_cindex_all': latest_val_cindex_all})

    results_latest_df.to_csv(f"{args.results_dir}/summary_latest_{str(args.param_code)}_{str(args.exp_code)}.csv",
                             index=False)

    csv_combine_pth = f'./summary_latest_val_in_train/{args.which_splits}/{args.model_type}/'
    os.makedirs(csv_combine_pth, exist_ok=True)
    results_latest_df.to_csv(f"{csv_combine_pth}/summary_latest_{str(args.param_code)}_{str(args.exp_code)}.csv",
                             index=False)
    print()
    print(results_latest_df, flush=True)
    print()


### Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--project_name', type=str, default='PAAD')
parser.add_argument('--yml_opt_path', type=str, default='protosurv')
args = parser.parse_args()
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args(args, configs_dir='configs')

    print('args.results_dir', args.results_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    seed_torch(args.seed)

    settings = {}
    for attr_name in dir(args):
        if "__" not in attr_name and '_get_' not in attr_name:
            settings[attr_name] = getattr(args, attr_name)

    ### Sets the absolute path of split_dir
    args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
    print("split_dir", args.split_dir)
    assert os.path.isdir(args.split_dir)
    settings.update({'split_dir': args.split_dir})

    print('\nLoad Dataset')
    dataset = Generic_MIL_Survival_Dataset(csv_path=args.csv_path,
                                           data_dir=args.data_dir,
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=False,
                                           patient_strat=False,
                                           label_col='survival_months',
                                           in_memory=args.in_memory,
                                           bar=args.bar,
                                           args_o=args
                                           )

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("################# Settings Finish ###################")
    print(flush=True)

    with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()
    # shutil.rmtree(args.results_dir,ignore_errors=True)

    start = timer()
    results = main(args)
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
