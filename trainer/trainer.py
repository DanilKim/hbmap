import numpy as np
import torch
import math
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.util import get_tile_bbox

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, grad_scaler=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.grad_scaler = grad_scaler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            img, mask = data['image'], data['mask']
            img, mask = img.to(self.device), mask.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.config['amp']['args']['enabled']):
                output = self.model(img)
                loss = 0
                for criterion in self.criterions:
                    loss += criterion(output, mask)
                #loss = torch.sum(torch.Tensor([criterion(output, mask) for criterion in self.criterions]))

            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, mask))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(img.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        image_size = self.config.data_loader.args.image_size
        tile_size = self.config.data_loader.args.tile_size
        tile_h = math.ceil( image_size / tile_size )
        with torch.no_grad():
            whole_output = torch.zeros(image_size, image_size)
            whole_mask = torch.zeros(image_size, image_size)
            cur_name = ''
            for batch_idx, data in enumerate(self.valid_data_loader):
                img, mask, name, tile_ids = data['image'], data['mask'], data['name'], data['tile_id']
                img, mask = img.to(self.device), mask.to(self.device)
                if cur_name != name:
                    whole_output = torch.zeros(image_size, image_size)
                    whole_mask = torch.zeros(image_size, image_size)

                output = self.model(img)
                loss = 0
                for criterion in self.criterions:
                    loss += criterion(output, mask)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for i, tid in enumerate(tile_ids):
                    y1, y2, x1, x2 = get_tile_bbox(image_size, tile_size, tid, False)
                    whole_output[ i, :, y1 : y2, x1 : x2 ] = output
                    whole_mask[ i, y1 : y2, x1 : x2 ] = mask

                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, mask))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
