import argparse
import os
import torch
import models
import dataloader
import json
import numpy as np


class Options():
    """
    This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    这部分参数来源于两部分：initialize中定义的，以及脚本文件中预先设置的
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str,
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--description', type=str, default=None)
        parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--save_dir', type=str, default='./results', help='logs and models are saved here')
        parser.add_argument('--shared_dir', type=str, default='./shared', help='shared are saved here')
        parser.add_argument('--cuda_benchmark', action='store_true', help='use torch cudnn benchmark')
        # model parameters
        parser.add_argument('--model', type=str, default='mmin',
                            help='chooses which model to use. [autoencoder | siamese | emotion_A]')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay when training')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier_normal | xavier_uniform | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012,
                            help='scaling factor for normal, xavier and orthogonal.')
        

        # dataset parameters
        parser.add_argument('--dataset_name', type=str, default='MOSEI', choices=['MOSEI','MOSI','MOSI2','SIMS','SIMSv2s','URFUNNY','VAD','CHERMA'],
                            help='chooses how datasets are loaded.')
        parser.add_argument('--input_dim_a', type=int, default=1024, help='acoustic input dim')
        parser.add_argument('--input_dim_t', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--text_context_length', type=int, default=2)
        parser.add_argument('--audio_context_length', type=int, default=1)
        parser.add_argument('--aligned', action='store_true', help='whether is aligned')
        
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--prefetch_factor', type=int, default=2)
        # parser.add_argument('--use_similarity', action='store_true')
        # parser.add_argument('--simi_return_mono', action='store_true')
        # parser.add_argument('--serial_batches', action='store_true',
        #                     help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        ## training parameter
        parser.add_argument('--train_mode', type=str, default='regression', choices=['regression','classification'])
        parser.add_argument('--optimizer_mode', type=str, default='adamw', choices=['adam','adamw'])
        
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=80,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--con_loss', type=str, default='supcon', choices=['supcon','acccon','uacccon','rnc'])

        # Train
        parser.add_argument('--bert_finetune', action='store_true', help='whether finetune BERT?')
        parser.add_argument('--bert_pretrain', type=str, default='bert-base-uncased', help='which bert to finetune')
        parser.add_argument('--output_dim', type=int, default=1, help='how many label types in this dataset')
        parser.add_argument('--qformer_heads', type=int, default=8)
        parser.add_argument('--qformer_layers', type=int, default=3)
        parser.add_argument('--query_token_len', type=int, default=4)
        parser.add_argument('--num_experts', type=int, default=8)
        parser.add_argument('--num_classes', type=int, default=7)
        parser.add_argument('--eval_batch_size', type=int, default=32)
        parser.add_argument('--n_epoch', type=int, default=50)
        parser.add_argument('--patience', type=int, default=2)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--shapri_dim', type=int, default=64)
        parser.add_argument('--shauni_dim', type=int, nargs='+', default=[128,128,128])
        parser.add_argument('--shauni_tr_dim', type=int, nargs='+', default=[64,64,64])
        parser.add_argument('--unique_dim', type=int, default=64)
        parser.add_argument('--shauni_task_dim', type=int, nargs='+')
        parser.add_argument('--mi_hidden_dim', type=int, nargs='+')
        parser.add_argument('--task_dim', type=int, default=64)
        # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
        parser.add_argument('--activation', type=str, default='relu')

        parser.add_argument('--bert_lr', type=float, default=1e-5, help='initial learning rate for adam')
        parser.add_argument('--mi_lr', type=float, default=1e-5, help='initial learning rate for adam')
        parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--milestonse', default=[50, 100], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10)')
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
        parser.add_argument('--random_seed', default=3407, type=int, help='# the random seed')

        # ---------END---------

        # expr setting 
        parser.add_argument('--best_epoch', type=int, default=-1)
        parser.add_argument('--run_idx', type=str, help='experiment number; for repeat experiment')
        self.isTrain = True
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        # print(model_name)
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_option_setter = dataloader.get_option_setter(opt.dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)  # 这个函数目前没东西，parser还是原来的parser

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        if opt.verbose:
            print(message)
        
        # save opt as txt file
        # file_name = os.path.join(opt.checkpoints_dir, '{}_opt.txt'.format(opt.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self, isTrain=True):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = isTrain  # train or test

        if opt.isTrain:
            opt.name = opt.model+'_'+opt.dataset_name.lower()
            if opt.description:
                opt.name = opt.name + '_' + opt.description
            # process opt.suffix
            if opt.suffix:
                suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
                opt.name = opt.name + suffix
                print("Expr Name:", opt.name)

        # checkpoint save dir
        opt.checkpoints_dir = os.path.join(opt.save_dir, opt.name, 'checkpoints')
        os.makedirs(opt.checkpoints_dir, exist_ok=True)
            
        # log save dir
        opt.log_dir = os.path.join(opt.save_dir, opt.name,'logs')
        os.makedirs(opt.log_dir, exist_ok=True)

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


def save_json(opt):
    dictionary = {}
    for k, v in sorted(vars(opt).items()):
        dictionary[k] = v

    save_path = os.path.join(opt.checkpoints_dir, '{}_opt.conf'.format(opt.phase))
    with open(save_path, 'w', encoding='UTF-8') as f:
        json.dump(dictionary, f, indent=4)