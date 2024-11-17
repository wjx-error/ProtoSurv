import math
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
import time
from utils.utils import Sampler_custom

from utils.core_funcs import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(datasets: tuple, args: Namespace, cur=0, max_node_num=20_0000, bar=True):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    if test_split != None:
        save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, f'splits_{cur}.csv'))
    else:
        save_splits((train_split, val_split), ['train', 'val'], os.path.join(args.results_dir, f'splits_{cur}.csv'))

    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    loss_fn = get_loss(args)
    print(args.bag_loss)
    print('loss_fn', type(loss_fn))

    reg_fn = None
    model_type = args.model_type

    model = get_model(args)
    print('model', type(model))

    gpu_ids = args.gpu_ids
    print('gpu_ids', gpu_ids)
    if isinstance(gpu_ids, list):
        torch.cuda.set_device(gpu_ids[0])
    else:
        torch.cuda.set_device(gpu_ids)
        gpu_ids = [gpu_ids]

    model = DataParallel(model, device_ids=gpu_ids)
    model = model.to(torch.device('cuda'))
    print('Done!')

    print('\nInit Loaders...')
    print('len dataset', len(train_split))
    if args.custom_sample:
        print('Use custom Sampler')
        print('args.batch_size', args.batch_size)
        train_batch_sampler = Sampler_custom(np.array(train_split.censorship), args.batch_size)
        print(type(train_batch_sampler))
        train_loader = DataListLoader(train_split, batch_sampler=train_batch_sampler,
                                      num_workers=args.num_workers, pin_memory=args.pin_memory)
    else:
        if args.bag_loss == 'cox' and args.alpha_surv == 0:
            print('Warning: cox loss and no alpha, may get nan loss!!!!!!!!!!!!!!!!')
        train_loader = DataListLoader(train_split, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=args.pin_memory, drop_last=True)

    val_censorship = val_split.censorship
    val_event_list = np.where(np.array(val_censorship) == 0)[0]
    val_censor_list = np.where(np.array(val_censorship) == 1)[0]
    print('val event list', len(val_event_list))
    print('val censor list', len(val_censor_list))

    batch_size = 1
    val_loader = DataListLoader(val_split, batch_size=batch_size, num_workers=args.num_workers, shuffle=False)
    if test_split != None:
        test_loader = DataListLoader(test_split, batch_size=batch_size, num_workers=args.num_workers,
                                     shuffle=False)

    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping_cindex(warmup=0, patience=15, stop_epoch=20, verbose=True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    print('Done!')

    for epoch in range(1, args.max_epochs + 1):
        st = time.time()
        try:
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes, args, writer, loss_fn,
                                reg_fn, args.lambda_reg, args.gc, max_node_num=max_node_num, model_type=model_type, bar=bar)
            print(f'Epoch: {epoch}, learning_rate:', optimizer.param_groups[0]['lr'])
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError):
                print('OutOfMemoryError')
            else:
                print(e)
            continue

        stop = validate_survival(epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn,
                                 reg_fn, args.lambda_reg, args.results_dir, cur=cur, max_node_num=max_node_num,
                                 model_type=model_type, bar=bar)
        if args.testing:
            validate_survival(epoch, model, test_loader, args.n_classes, None, None, loss_fn,
                              reg_fn, args.lambda_reg, args.results_dir, cur=cur, max_node_num=max_node_num,
                              model_type=model_type, bar=bar, mode='Test')
        scheduler.step()
        ed = time.time()
        print(f"Epoch {epoch} cost {ed - st}s", flush=True)
        print()
        if stop:
            print('early stop break', flush=True)
            break
        gc.collect()

    torch.save(model.state_dict(), os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt"))
    model.load_state_dict(
        torch.load(os.path.join(args.results_dir, f"s_{cur}_maxcindex_checkpoint.pt"), map_location='cpu'))
    model = model.to(torch.device('cuda'))

    if test_split != None:
        print('cal val and test split')
        results_val_dict, val_cindex, val_c_index_all, cnt = summary_survival(model, val_loader, bar=bar)
        print('Val c-Index_censored: {:.4f}, Val c-Index_all: {:.4f}'.format(val_cindex, val_c_index_all), flush=True)
        results_test_dict, test_cindex, test_c_index_all, cnt = summary_survival(model, test_loader, bar=bar)
        print('Test c-Index_censored: {:.4f}, Test c-Index_all: {:.4f}'.format(test_cindex, test_c_index_all),
              flush=True)
        writer.close()
        return (results_val_dict, val_cindex, val_c_index_all), (results_test_dict, test_cindex, test_c_index_all)
    else:
        print('cal val split')
        results_val_dict, val_cindex, val_c_index_all, cnt = summary_survival(model, val_loader, bar=bar)
        print('Val c-Index_censored: {:.4f}, Val c-Index_all: {:.4f}'.format(val_cindex, val_c_index_all, ), flush=True)
        writer.close()
        return (results_val_dict, val_cindex, val_c_index_all)
