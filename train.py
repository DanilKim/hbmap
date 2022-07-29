import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,5"

import argparse
import collections
from random import shuffle
import torch
import numpy as np
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    n_classes = config['arch']['args']['n_classes']
    data_loader = config.init_obj('data_loader', module_data, n_classes=n_classes)
    valid_data_loader = config.init_obj('valid_data_loader', module_data, n_classes=n_classes)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    seg_params, cls_params = model.define_params_for_optimizer()

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterions = {
        "cls": getattr(module_loss, config['cls_loss']),
        "seg": getattr(module_loss, config['seg_loss'])
    }
    metrics = {
        "cls": [getattr(module_metric, met) for met in config['cls_metrics']],
        "seg": [getattr(module_metric, met) for met in config['seg_metrics']]
    }

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    
    optimizers = {
        "cls": config.init_obj('cls_optimizer', torch.optim, cls_params),
        "seg": config.init_obj('seg_optimizer', torch.optim, seg_params)
    }
    lr_schedulers = {
        "cls": config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizers["cls"]),
        "seg": config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizers["seg"]),
    }

    trainer = Trainer(model, criterions, metrics, optimizers,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_schedulers)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
