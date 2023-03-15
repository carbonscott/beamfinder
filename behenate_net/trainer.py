import logging
import os
import tqdm
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class ConfigTrainer:
    path_chkpt   = None
    num_workers  = 4
    batch_size   = 64
    max_epochs   = 10
    lr           = 0.001
    tqdm_disable = False

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Trainer \___")
        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class Trainer:
    def __init__(self, model, dataset, config):
        self.model   = model
        self.dataset = dataset
        self.config  = config

        # Load data to gpus if available...
        self.device = device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.model = torch.nn.DataParallel(self.model).to(self.device, dtype = torch.float)

        return None


    def save_checkpoint(self, epoch = None):
        DRCCHKPT = "chkpts"
        drc_cwd = os.getcwd()
        prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
        if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
        fl_chkpt = self.config.timestamp

        if epoch is not None:
            fl_chkpt += f'.epoch_{epoch}'
            fl_chkpt += f'.chkpt'

        path_chkpt = os.path.join(prefixpath_chkpt, fl_chkpt)

        # Hmmm, DataParallel wrappers keep raw model object in .module attribute
        model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"SAVE - {path_chkpt}")
        torch.save(model.state_dict(), path_chkpt)


    def train(self, saves_checkpoint = True, epoch = None, returns_loss = False, logs_batch_loss = False):
        """ The training loop.  """
        # Load model and training configuration...
        # Optimizer can be reconfigured next epoch
        model, config = self.model, self.config
        model_raw     = model.module if hasattr(model, "module") else model
        optimizer     = model_raw.configure_optimizers(config)

        # Train an epoch...
        model.train()
        dataset = self.dataset
        loader_train = DataLoader( dataset, shuffle     = config.shuffle, 
                                            pin_memory  = config.pin_memory, 
                                            batch_size  = config.batch_size,
                                            num_workers = config.num_workers )

        # Train each batch...
        losses_epoch = []
        batch_list = tqdm.tqdm(enumerate(loader_train), total = len(loader_train), disable = config.tqdm_disable)
        for batch_id, batch in batch_list:
            batch_img, batch_center, batch_metadata = batch

            # [B, B] -> (2, B) -> (B, 2)
            batch_center = torch.cat(batch_center, dim = 0).view(2, -1).transpose(0, 1)

            batch_img    = batch_img.to(device = self.device, dtype = torch.float)
            batch_center = batch_center.to(device = self.device, dtype = torch.float)

            loss = self.model.forward(batch_img, batch_center)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.cpu().detach().numpy()
            losses_epoch.append(loss_val)

            if logs_batch_loss:
                logger.info(f"MSG - epoch {epoch}, batch {batch_id:d}, loss {loss_val:.8f}")

        loss_epoch_mean = np.mean(losses_epoch)
        logger.info(f"MSG - epoch {epoch}, loss mean {loss_epoch_mean:.8f}")

        return loss_epoch_mean if returns_loss else None
