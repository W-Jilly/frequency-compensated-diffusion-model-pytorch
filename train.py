import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import copy
import random

seed = 6666
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.cache = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n

        self.cache.append(self.val)
        if len(self.cache) >= 20: self.cache = self.cache[1:]
        self.avg = np.mean(self.cache)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def adjust_learning_rate(change_idx, optimizer):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

        lr = lr * (0.7 ** change_idx)

        param_group['lr'] = lr

    logger.info("Current lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/framework_da.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    change_sizes = opt["change_sizes"]

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    logger.info("change rate:" + "".join(["{}:{} ".format(k, v) for k, v in change_sizes.items()]))

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    # ave
    ave_loss = AverageMeter()

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if current_step == 0:
        change_size_idx = 0
    else:
        change_size_idx = 0
        try:
            while current_step >= int(
                    float(list(change_sizes.keys())[change_size_idx]) * n_iter) and change_size_idx < len(
                    list(change_sizes.keys())):
                change_size_idx += 1
        except:
            pass
        change_size_idx -= 1

    while current_step < n_iter:

        # reset train_loader
        if current_step >= int(
                float(list(change_sizes.keys())[change_size_idx]) * n_iter) and change_size_idx < len(
                list(change_sizes.keys())):
            logger.info('reset train_loader')
            resize_resolu = change_sizes[list(change_sizes.keys())[change_size_idx]]
            train_dataset_opt = copy.deepcopy(opt['datasets']['train'])

            train_dataset_opt["l_resolution"], train_dataset_opt["r_resolution"] = resize_resolu, resize_resolu

            logger.info('reset train_loader: l_resolution:{}, r_resolution:{}, batch_size:{}'.format(
                train_dataset_opt["l_resolution"], train_dataset_opt["r_resolution"],
                train_dataset_opt["batch_size"]))

            train_set = Data.create_dataset(train_dataset_opt, 'train')
            train_loader = Data.create_dataloader(train_set, train_dataset_opt, 'train')

            logger.info('reset train_loader finished .')

            adjust_learning_rate(change_size_idx, diffusion.optG)

            change_size_idx += 1

        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters(current_step)
            # log
            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    ave_loss.update(v)
                    message += '{:s}: {:.4e} ({:.4e})'.format(k, v, ave_loss.avg)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics(logs)

            if current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

                if wandb_logger and opt['log_wandb_ckpt']:
                    wandb_logger.log_checkpoint(current_epoch, current_step)

            if current_step >= int(
                    float(list(change_sizes.keys())[change_size_idx]) * n_iter) and change_size_idx < len(
                list(change_sizes.keys())):
                break

        if wandb_logger:
            wandb_logger.log_metrics({'epoch': current_epoch - 1})

    # save model
    logger.info('End of training.')
