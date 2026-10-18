import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from .networks import tools


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = opt.checkpoints_dir  # save all the checkpoints to save_dir
        self.log_dir = opt.log_dir
        self.save_shared_dir = opt.shared_dir
        if opt.cuda_benchmark: # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.metric = 0  # used for learning rate policy 'plateau'


    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, logger=None):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [tools.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            for name in self.model_names:
                init_weight=True
                if name.lower().find('bert') >= 0:
                    init_weight=False
                
                net = getattr(self, 'net' + name)
                if logger !=None:
                    logger.info('initialize network:%s' % name)
                else:
                    print('initialize network:%s' % name)
                net = tools.init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids,init_weight=init_weight)
                setattr(self, 'net' + name, net)
        else:
            self.eval()

        self.print_networks(verbose=opt.verbose, logger=logger)
        self.post_process()

    def cuda(self):
        assert(torch.cuda.is_available())
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.to(self.gpu_ids[0])
            net = torch.nn.DataParallel(net, self.gpu_ids)  # multi-GPUs

    def eval(self):
        """Make models eval mode during test time"""
        self.isTrain = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    def train(self):
        """Make models back to train mode after test time"""
        self.isTrain = True
        model_list = self.model_names
        if getattr(self,'fsf',None) != None and self.fsf == True:
            model_list = self.second_stage_model_names
        for name in model_list:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def update_learning_rate(self, logger=None, metric=None):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy in ['reduce_on_plateau']:
                scheduler.step(metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)
        if logger != None:
            logger.info('learning rate = %.7f' % lr)


    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def get_criterion_loss(self, pred, label):
        pass

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            # print("model_name is:",name)
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                # print('save_path is:', save_path)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    # print(type(self.gpu_ids),self.gpu_ids)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch, logger=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if logger!=None:
            logger.info('loading the model from %s / %s_net' % (self.save_dir, epoch))
        else:
            print('loading the model from %s / %s_net' % (self.save_dir, epoch))
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                state_dict = torch.load(load_path, map_location=self.device, weights_only=False)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)
    
    def load_networks_cv(self, folder_path):
        """Load all the networks from cv folder.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        checkpoints = list(filter(lambda x: x.endswith('.pth'), os.listdir(folder_path)))
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = list(filter(lambda x: x.split('.')[0].endswith('net_'+name), checkpoints))
                print('load_filename is:', load_filename)
                # assert len(load_filename) == 1, 'In folder: {}, Exists file {}'.format(folder_path, load_filename)
                if len(load_filename) == 1:
                    load_filename = load_filename[0]
                    load_path = os.path.join(folder_path, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=self.device, weights_only=False)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    net.load_state_dict(state_dict)
                else:
                    continue
    def load_networks_epoch(self, folder_path, epoch):
        """Load all the networks from cv folder.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        checkpoints = list(filter(lambda x: x.endswith('.pth'), os.listdir(folder_path)))
        prefix = str(epoch)+'_net_'
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = list(filter(lambda x: x.split('.')[0].endswith(prefix+name), checkpoints))
                print('load_filename is:', load_filename)
                # assert len(load_filename) == 1, 'In folder: {}, Exists file {}'.format(folder_path, load_filename)
                if len(load_filename) == 1:
                    load_filename = load_filename[0]
                    load_path = os.path.join(folder_path, load_filename)
                    net = getattr(self, 'net' + name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    state_dict = torch.load(load_path, map_location=self.device, weights_only=False)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    net.load_state_dict(state_dict)
                else:
                    continue

    def print_networks(self, verbose=False, logger=None):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('--------------- Networks initialized ------------------')
        total_params = 0
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    if logger != None:
                        logger.info(net)
                    else:
                        print(net)
                if logger != None:
                    logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                else:
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
            total_params += num_params
        if logger != None:
            logger.info('Total number of Model : %d' % (total_params))
        else:
            print('Total number of Model : %d' % (total_params))
        print('---------------------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def post_process(self):
        pass