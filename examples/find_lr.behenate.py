#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
import random
import tqdm
import pickle

import matplotlib              as mpl
import matplotlib.pyplot       as plt
import matplotlib.colors       as mcolors
import matplotlib.patches      as mpatches
import matplotlib.transforms   as mtransforms
import matplotlib.font_manager as font_manager

from behenate_net.aug import Pad,                  \
                             Crop,                 \
                             RandomCenterCropZoom, \
                             Resize,               \
                             RandomShift,          \
                             RandomRotate,         \
                             RandomPatch

from behenate_net.dataset   import BeHenateDataset
from behenate_net.model     import ConfigModel, BeHenataNet
from behenate_net.trainer   import ConfigTrainer, Trainer
from behenate_net.validator import ConfigValidator, Validator
from behenate_net.utils     import EpochManager, split_dataset, set_seed, init_logger, MetaLog

seed = 0
set_seed(seed)

batch_size = 200
lr_exponents = np.linspace(-6, -1, 100)
frac_train = 0.5
frac_validate = 0.5

size_sample = 200
size_img_y, size_img_x = (200, 200)

size_pad       = 2000
size_patch     = 20
frac_shift_max = 0.4
angle_max      = 359

uses_frac_center = True

timestamp_prev = "2023_0317_2047_44" # "2023_0316_1002_21"   # ...or None
epoch = 277
fl_chkpt = None if timestamp_prev is None else f"{timestamp_prev}.epoch_{epoch}.chkpt"

timestamp = init_logger(log_name = 'train', returns_timestamp = True)
print(timestamp)

comments = f"""
time prev      : {timestamp_prev}
frac_train     : {frac_train}
size_sample    : {size_sample}
size_pad       : {size_pad}
RandomPatch    : {size_patch}
frac_shift_max : {frac_shift_max}
angle_max      : {angle_max}
"""
metalog = MetaLog( comments = comments )
metalog.report()

path_pickle = "beam_center.pickle"
with open(path_pickle, 'rb') as handle:
    data_list = pickle.load(handle)

# Split data...
data_train   , data_val_and_test = split_dataset(data_list        , frac_train   , seed = None)
data_validate, data_test         = split_dataset(data_val_and_test, frac_validate, seed = None)

normalizes_data = True
trans_list = (
    Pad(size_y = size_pad, size_x = size_pad),
    Resize(size_img_y, size_img_x),
    RandomRotate(angle_max = angle_max),
    RandomShift(frac_shift_max, frac_shift_max),
    RandomCenterCropZoom(trim_factor_max = 0.2),
    ## RandomPatch(num_patch = 10, size_patch_y = size_patch, size_patch_x = size_patch, var_patch_y = 0.2, var_patch_x = 0.2),
)
dataset_train = BeHenateDataset( data_list          = data_train,
                                 size_sample        = size_sample,
                                 trans_list         = trans_list,
                                 normalizes_data    = normalizes_data,
                                 uses_frac_center   = uses_frac_center,
                                 prints_cache_state = False,
                               )
dataset_train.cache_dataset()

dataset_validate = BeHenateDataset( data_list          = data_validate,
                                    size_sample        = size_sample // 4,
                                    trans_list         = trans_list,
                                    normalizes_data    = normalizes_data,
                                    uses_frac_center   = uses_frac_center,
                                    prints_cache_state = False,
                                  )
dataset_validate.cache_dataset()

config_model = ConfigModel( size_y = size_img_y, size_x = size_img_x, isbias = True )
model = BeHenataNet(config_model)
model.init_params(fl_chkpt = fl_chkpt)

# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( timestamp    = timestamp,
                              num_workers  = 2,
                              batch_size   = batch_size,
                              pin_memory   = True,
                              shuffle      = False,
                              lr           = None,
                              tqdm_disable = True)
trainer = Trainer(model, dataset_train, config_train)

# [[[ VALIDATOR ]]]
# Config the validator...
config_validator = ConfigValidator( num_workers  = 2,
                                    batch_size   = batch_size,
                                    pin_memory   = True,
                                    shuffle      = False,
                                    lr           = None,
                                    tqdm_disable = True)
validator = Validator(model, dataset_validate, config_validator)

# Scan learning rate...
loss_by_lr_list = []
for enum_idx, lr_exponent in enumerate(tqdm.tqdm(lr_exponents, disable=False)):
    lr = 10 ** lr_exponent
    trainer.config.lr = lr
    validator.config.lr = lr

    ## # [[[ TRAIN EPOCHS ]]]
    ## loss_train_hist    = []
    ## loss_validate_hist = []
    ## loss_min_hist      = []

    # [[[ EPOCH MANAGER ]]]
    epoch_manager = EpochManager( trainer   = trainer,
                                  validator = validator, )

    max_epochs = 1
    freq_save = 5
    for epoch in range(max_epochs):
        loss_train, loss_validate, loss_min = epoch_manager.run_one_epoch(epoch = epoch, returns_loss = True)

        ## loss_train_hist.append(loss_train)
        ## loss_validate_hist.append(loss_validate)
        ## loss_min_hist.append(loss_min)

        # if epoch % freq_save == 0: 
        #     epoch_manager.save_model_parameters()
        #     epoch_manager.save_model_gradients()
        #     epoch_manager.save_state_dict()
    loss_by_lr_list.append([float(lr_exponent), float(loss_train), float(loss_validate), float(loss_min)])

    if enum_idx % 50 == 0:
        fl_pickle = "find_lr.pickle"
        with open(fl_pickle, 'wb') as fh:
            pickle.dump(loss_by_lr_list, fh)

fl_pickle = "find_lr.pickle"
with open(fl_pickle, 'wb') as fh:
    pickle.dump(loss_by_lr_list, fh)
