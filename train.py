import os
import time
import numpy as np
from get_opts import Options, save_json
from dataloader import create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from utils.metrics import Metrics
from utils.util import set_random_seed,plot_confusion_matrics
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import torch
from torch.nn import MSELoss
from random import random
from collections import OrderedDict

import pickle
from tqdm import tqdm

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.cur = None
        self.count = 0

    def update(self, val, n=1):
        if self.val == None:
            self.val={}
            for key,value in val.items():
                self.val[key]=[value]
        else:
            for key,value in val.items():
                self.val[key].append(value)
        self.cur = val
        self.count += n
    
    def avg(self):
        res = OrderedDict()
        for key,value in self.val.items():
            res[key]= sum(value) / self.count

        return res


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval(model, val_iter, metrics, train_mode='regression', is_save=False, phase='val'):
    model.eval()
    total_pred = []
    total_label = []
    total_loss = 0.
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.detach().cpu()
        if train_mode == 'regression':
            label = data['regression_labels'].detach().cpu()
        else:
            label = data['classification_labels'].detach().cpu()
        total_pred.append(pred)
        total_label.append(label)
        loss = model.get_criterion_loss(pred, label)
        total_loss += loss.item()

    total_loss = total_loss / len(val_iter)

    # calculate metrics
    total_pred = torch.cat(total_pred,dim=0)
    total_label = torch.cat(total_label, dim=0)

    eval_res = metrics(total_pred, total_label)
    
    model.train()

    # save test results
    if is_save:
        save_dir = model.log_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred.numpy())
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label.numpy())

    eval_res[f'{phase}_loss']=total_loss
    return eval_res


def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not any(checkpoint.startswith(str(e) + "_") for e in list(store_epoch.values())) and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))


if __name__ == '__main__':
    opt = Options().parse()  # get training options
    
    logger_path = os.path.join(opt.log_dir, str(opt.cvNo))  # get logger path
    os.makedirs(logger_path, exist_ok=True) # make sure logger path exists

    suffix = '_'.join([opt.model, opt.dataset_name])  # get logger suffix
    logger = get_logger(logger_path, suffix)  # get logger
    dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, type=['train', 'test', 'test'])

    # 6231  731
    # seed = [7154,1387,-1]  # -1 means code needs random set
    seed = [7154,3585,8757,5110,-1]  # -1 means code needs random set

    opt.random_seed = set_random_seed(seed=seed[opt.cvNo-1])
    logger.info('Current global seed is {}'.format(opt.random_seed))

    dataset_size = len(dataset)  # get the number of images in the dataset.
    epoch_iter_len = dataset_size // opt.batch_size
    opt.num_training_steps = epoch_iter_len * (opt.niter + opt.niter_decay + 1-opt.epoch_count)
    
    # opt.num_group = [1, 2, 4, 8, 12, 16, 24, 32]
    opt.num_group = [opt.query_token_len for _ in range(opt.num_experts)]

    dataparams =  dataset._get_dataparams()
    for key, value in dataparams.items():
        setattr(opt, key, value)
    
    save_json(opt)

    logger.info('The number of training samples = %d' % dataset_size)
    logger.info("Effective parameters:")
    for key in sorted(opt.__dict__):
        logger.info("  <<< {}: {}".format(key, opt.__dict__[key]))

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt, logger=logger)  # regular setup: load and print networks; create schedulers
    metrics = Metrics(train_mode=opt.train_mode).getMetics(opt.dataset_name)
    total_iters = 0  # the total number of training iterations
    big_val= 1000.
    if opt.dataset_name in ['MOSEI','MOSI','MOSI2']:
        select_metric = {'val_loss':1,
                        'Has0_acc_2':-1,
                        'Has0_F1_score':-1,
                        'Non0_acc_2':-1,
                        'Non0_F1_score':-1,
                        'Mult_acc_5':-1,
                        'Mult_acc_7':-1,
                        'MAE':1,
                        'Corr':-1}   # -1:max 1:min
    elif opt.dataset_name in ['SIMS','SIMSv2s']:
        select_metric = {'val_loss':1,
                        'Mult_acc_2':-1,
                        'Mult_acc_3':-1,
                        'Mult_acc_5':-1,
                        'F1_score':-1,
                        'MAE':1,
                        'Corr':-1}   # -1:max 1:min
    elif opt.dataset_name in ['URFUNNY']:
        select_metric = {'val_loss':1,
                        'Acc2':-1,
                        'Weighted_F1':-1,
                        'Average_F1':-1}
    elif opt.dataset_name in ['VAD']:
        select_metric = {'val_loss':1,
                        'Acc2':-1,
                        'micro_F1':-1}
    else:
        select_metric = {'val_loss':1,
                        'Acc':-1}
    result_recorder = ResultRecorder(os.path.join(opt.log_dir, 'result.csv'),total_cv=3*len(select_metric))  # init result recoreder
    
    best_eval_epoch = {k:-1 for k,v in select_metric.items()}  # record the best eval epoch
    best_metric = {k:v*big_val for k,v in select_metric.items()}

    # 修改多线程的tensor方式为file_system，以解决打开文件的数量受限的问题
    # torch.multiprocessing.set_sharing_strategy('file_system')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        total_loss = 0
        losses = AverageMeter()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1  # opt.batch_size
            epoch_iter += 1
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)  # calculate loss functions, get gradients, update network weights
            losses.update(model.get_current_losses())
            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                avg_loss = losses.avg()
                losses.reset()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) +' iter:{}/{}'.format(epoch_iter, epoch_iter_len)+ ' loss ' +
                            ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**avg_loss))
                # for loss in losses.values():
                #     total_loss += loss
            iter_data_time = time.time()
            if opt.lr_policy in ['linear_with_warmup','cosine_with_warmup']:
                # update learning rates at the end of every iter.
                if total_iters % opt.print_freq == 0:
                    model.update_learning_rate(logger)
                else:
                    model.update_learning_rate()


        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            logger.info('save to: %s' % (model.save_dir))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        

        # eval val set
        eval_results = eval(model, val_dataset, metrics, train_mode=opt.train_mode, is_save=True)
        logger.info('Val result of epoch %d / %d' % (epoch, opt.niter + opt.niter_decay))
        for key, val in eval_results.items():
            logger.info('Val: %s:%.4f' %(key,val))

        # record epoch with best result
        for key,flag in select_metric.items():
            isSave=False
            if flag <= 0:
                isSave= eval_results[key] > best_metric[key]
            else:
                isSave= eval_results[key] < best_metric[key]
            
            if isSave:
                best_metric[key] = eval_results[key]
                best_eval_epoch[key]=epoch
        
        for k, v in best_metric.items():
            logger.info('Cur Val best %15s in epoch %d, best_metric= %.4f' % (k, best_eval_epoch[k], best_metric[k]))

        # update learning rates at the end of every epoch.
        if opt.lr_policy not in ['linear_with_warmup','cosine_with_warmup']:
            if opt.lr_policy in ['reduce_on_plateau']:
                model.update_learning_rate(logger,metric=eval_results['val_loss'])
            else:
                model.update_learning_rate(logger)


        if epoch % 10 == 0 and epoch!=opt.niter + opt.niter_decay:
            for k,signal in select_metric.items():
                logger.info('Loading the best model based on the metric:%s epoch=%d' % (k,best_eval_epoch[k]))
                model.load_networks(best_eval_epoch[k], logger=logger)
                # start = time.time()
                # for item in tqdm(range(100)):
                test_result = eval(model, tst_dataset, metrics, train_mode=opt.train_mode, is_save=True, phase='test')
                # end = time.time()
                # logger.info('Test time cost: %.2f s' % (end - start))
                if signal <= 0:
                    adj='highest'
                else:
                    adj='lowest'
                logger.info('*'*154)
                logger.info('Choosing the %s %s = %.4f' % (adj, k, best_metric[k]))
                logger.info(' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), test_result.keys())).format(**test_result))
                logger.info('*'*154)
            # model.load_networks(epoch, logger=logger)  # reload the latest model for next epoch

    # print best eval result
    for k, v in best_metric.items():
        logger.info('Best eval epoch %d found based on %15s : %.4f' % (best_eval_epoch[k], k, best_metric[k]))
    logger.info('Log save to path: %s' % (logger_path))
    
    # test
    for k,signal in select_metric.items():

        logger.info('Loading the best model based on the metric:%s epoch=%d' % (k,best_eval_epoch[k]))
        model.load_networks(best_eval_epoch[k], logger=logger)
        test_result = eval(model, tst_dataset, metrics, train_mode=opt.train_mode, is_save=True, phase='test')
        if signal <= 0:
            adj='highest'
        else:
            adj='lowest'
        logger.info('*'*154)
        logger.info('Choosing the %s %s = %.4f' % (adj, k, best_metric[k]))
        logger.info(' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), test_result.keys())).format(**test_result))
        logger.info('*'*154)
        result_recorder.write_result_to_tsv(test_result, cvNo=opt.cvNo)

    clean_chekpoints(str(opt.cvNo), best_eval_epoch)