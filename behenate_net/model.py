import os
import torch
import torch.nn as nn
from .utils import NNSize, TorchModelAttributeParser

from functools import reduce

import logging

logger = logging.getLogger(__name__)

class ConfigModel:

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Encoder \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class BeHenataNet(nn.Module):
    """ ... """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        bias           = config.isbias

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 32 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 64 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 3...
            nn.Conv2d( in_channels  = 64,
                       out_channels = 128,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 128 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 4...
            nn.Conv2d( in_channels  = 128,
                       out_channels = 128,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 128 ),
            nn.PReLU(),

            # CNN motif 5...
            nn.Conv2d( in_channels  = 128,
                       out_channels = 4,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 4 ),
            nn.PReLU(),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        ## # Define the residual transform...
        ## self.residual_linear = nn.Linear(
        ##     in_features  = size_y * size_x,
        ##     out_features = self.feature_size,
        ##     bias = True,
        ## )

        # Define the embedding layer...
        self.fc = nn.Sequential(
            nn.Linear( in_features  = self.feature_size,
                       out_features = 256,
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 256,
                       out_features = 16,
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 16,
                       out_features = 2,
                       bias         = bias),
        )


    def init_params(self, fl_chkpt = None):
        # Initialize weights or reuse weights from a timestamp...
        def init_weights(module):
            # Initialize conv2d with Kaiming method...
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

                # Set bias zero since batch norm is used...
                module.bias.data.zero_()

        if fl_chkpt is None:
            self.apply(init_weights)
        else:
            drc_cwd          = os.getcwd()
            DRCCHKPT         = "chkpts"
            prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
            path_chkpt_prev  = os.path.join(prefixpath_chkpt, fl_chkpt)
            self.load_state_dict(torch.load(path_chkpt_prev))


    def predict(self, batch_img):
        ## batch_img_copy = batch_img

        # Feature extraction with skip connection...
        ## batch_img = self.feature_extractor(batch_img)
        batch_img = self.feature_extractor(batch_img)
        batch_img = batch_img.view(-1, self.feature_size)
        ## batch_img += self.residual_linear(batch_img_copy.view(len(batch_img),-1))
        batch_center_pred = self.fc(batch_img)

        return batch_center_pred


    def forward(self, batch_img, batch_center):
        batch_center_pred = self.predict(batch_img)

        batch_center_diff = batch_center_pred - batch_center
        batch_center_diff = torch.abs(batch_center_diff)    # L1
        ## batch_center_diff *= batch_center_diff    # L2
        loss = torch.mean(batch_center_diff, dim = 1)

        return loss.mean()


    def configure_optimizers(self, config_train):
        optimizer = torch.optim.Adam(self.parameters(), lr = config_train.lr)

        return optimizer
