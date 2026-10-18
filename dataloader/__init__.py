"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_name dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
import numpy as np
from dataloader.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from pytorch_metric_learning import samplers
from dataloader.MOSEI_Dataset import MOSEIDataset
from dataloader.MOSI_Dataset import MOSIDataset

def MOSIDataloader(name, batch_size=None, use_sampler=False, use_similarity=False, simi_return_mono=False, shuffle=True,
                   num_workers=2,
                   prefetch_factor=2):
    if batch_size is None:
        print('batch size not defined')
        return
    dataset = MOSIDataset(name, use_similarity=use_similarity, simi_return_mono=simi_return_mono)
    sampler = None
    drop_last = False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=drop_last, timeout=0,
                      worker_init_fn=None, prefetch_factor=prefetch_factor,
                      persistent_workers=False)

def MOSEIDataloader(name, batch_size=None, use_sampler=False, use_similarity=False, simi_return_mono=False,
                    shuffle=True,
                    num_workers=2,
                    prefetch_factor=2):
    if batch_size is None:
        print('batch size not defined')
        return
    dataset = MOSEIDataset(name, use_similarity=use_similarity, simi_return_mono=simi_return_mono)
    sampler = None
    drop_last = False
    if use_sampler:
        shuffle = False
        drop_last = True
        sampler = samplers.MPerClassSampler(labels=dataset.data['regression_labels'],
                                            m=1,
                                            batch_size=None,
                                            # length_before_new_iter=len(dataset)
                                            length_before_new_iter=batch_size * 21
                                            )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=drop_last, timeout=0,
                      worker_init_fn=None, prefetch_factor=prefetch_factor,
                      persistent_workers=False)

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "dataloader." + dataset_name + "_Dataset"
    datasetlib = importlib.import_module(dataset_filename)  # 动态导入模块，模块名不用加.py

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset' # replace 替换子串，从左到右执行一次
    for name, cls in datasetlib.__dict__.items():
        # print("name is ", name) # name是data里的方法？
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):    # issubclass 用来判断两个类是否相等或为继承关系
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataloader given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    return data_loader


def create_dataset_with_args(opt, **kwargs):
    """Create two dataloader given the option, dataset may have additional args.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_split_dataset
        >>> dataset = create_dataset(opt, set_name=['trn', 'val', 'tst'])
        This will create 3 datasets, each one get different parameters
        for the specific dataset class, __init__ func must get one parameter
        eg: dataset.__init__(self, set_name='trn'): ....
    """
    _kwargs = []
    for key in kwargs:
        value = kwargs[key]
        if not isinstance(value, (list, tuple)):
            value = [value]
        lens = len(value)
        _kwargs += list(map(lambda x: {}, range(lens))) if len(_kwargs) == 0 else []
        for i, v in enumerate(value):
            _kwargs[i][key] = v 

    dataloaders = tuple(map(lambda x: CustomDatasetDataLoader(opt, **x), _kwargs))
    return dataloaders if len(dataloaders) > 1 else dataloaders[0]


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, **kwargs):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_name]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_name)
        self.dataset = dataset_class(opt, **kwargs)
        print("dataset [%s] on the [%s] set was successfully created." % (type(self.dataset).__name__, self.dataset.dataset_type))
        shuffle=False
        if self.dataset.dataset_type == 'train':
            shuffle=True
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=shuffle,
            num_workers=int(opt.num_threads),
            drop_last=False,
            prefetch_factor=12
        )

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def _get_dataparams(self):
        return self.dataset._get_dataparams()

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

def collate_fn(batch):
    print(batch)
    pass

def _init_fn(random_seed):
    np.random.seed(random_seed)