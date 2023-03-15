import logging
import os
import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class ConfigValidator:
    path_chkpt   = None
    num_workers  = 4
    batch_size   = 64
    max_epochs   = 10
    lr           = 0.001
    tqdm_disable = False

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Validator \___")
        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class Validator:
    def __init__(self, model, dataset, config):
        self.model   = model
        self.dataset = dataset
        self.config  = config

        # Load data to gpus if available...
        self.device = device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.model = torch.nn.DataParallel(self.model).to(self.device, dtype = torch.float)

        return None


    def validate(self, saves_checkpoint = True, epoch = None, returns_loss = False, logs_batch_loss = False):
        """ The validation loop.  """
        # Load model and validation configuration...
        # Optimizer can be reconfigured next epoch
        model, config = self.model, self.config
        model_raw     = model.module if hasattr(model, "module") else model
        optimizer     = model_raw.configure_optimizers(config)

        # Validate an epoch...
        model.eval()
        dataset = self.dataset
        loader_validate = DataLoader( dataset, shuffle     = config.shuffle, 
                                               pin_memory  = config.pin_memory, 
                                               batch_size  = config.batch_size,
                                               num_workers = config.num_workers )

        # Validate each batch...
        losses_epoch = []
        batch_list = tqdm.tqdm(enumerate(loader_validate), total = len(loader_validate), disable = config.tqdm_disable)
        for batch_id, batch in batch_list:
            batch_img, batch_center, batch_metadata = batch

            # [B, B] -> (2, B) -> (B, 2)
            batch_center = torch.cat(batch_center, dim = 0).view(2, -1).transpose(0, 1)

            batch_img    = batch_img.to(device = self.device, dtype = torch.float)
            batch_center = batch_center.to(device = self.device, dtype = torch.float)

            with torch.no_grad():
                loss = self.model.forward(batch_img, batch_center)

            loss_val = loss.cpu().detach().numpy()
            losses_epoch.append(loss_val)

            if logs_batch_loss:
                logger.info(f"MSG - epoch {epoch}, batch {batch_id:d}, loss {loss_val:.8f}")

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        return loss_epoch_mean if returns_loss else None

