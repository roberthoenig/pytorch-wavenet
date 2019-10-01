 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from modules.residual_stack import ResidualStack
from modules.conv1d_builder import Conv1DBuilder
from error_handling.console_logger import ConsoleLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from wavenet_utils import augmented_mfcc

class MFCC():
    def __init__(self, winlen, winstep, numcep, sampling_rate=16000):
        self.winlen = winlen
        self.winstep = winstep
        self.numcep = numcep
        self.sampling_rate = sampling_rate
    
    def __call__(self, data):
        '''
        data: (n_batches, length)
        returns: (n_batches, n_channels, downsampled_length)
        '''
        data = data - 128  # TODO: un-hardcode
        mfccs_out = [augmented_mfcc(
                x.squeeze().cpu().numpy(),
                winlen=self.winlen/self.sampling_rate,
                winstep=self.winstep/self.sampling_rate,
                numcep=self.numcep,
                appendEnergy=True
            ).transpose() for x in data.split(1, dim=0)]
        torch_mfccs_out = torch.stack([torch.tensor(x, dtype=torch.float).to(data.device) for x in mfccs_out])
        torch.set_printoptions(sci_mode=False)
        return torch_mfccs_out

class MFCCConvolutionalEncoder(nn.Module):
    
    def __init__(self, num_hiddens, num_outs, num_residual_layers, num_residual_hiddens,
        use_kaiming_normal, features_filters, winlen, winstep, numcep, verbose=False):

        super().__init__()

        self._mfcc = MFCC(winlen=winlen, winstep=winstep, numcep=numcep)

        self.batch_norm = nn.BatchNorm1d(numcep*3, eps=1e-05)

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            # padding=2
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal,
            init_dilation = 1,
            dilation_base = 1
        )

        """
        1 postprocessing convolution to obtain the desired num_outs
        channels.
        """
        self._conv_out = nn.Conv1d(
            in_channels=num_hiddens,
            out_channels=num_outs,
            kernel_size=3,
            padding=1
        )
        
        self._verbose = verbose

    def forward(self, inputs):
        if self._verbose:
            ConsoleLogger.status('inputs size: {}'.format(inputs.size()))

        mfcc_inputs = self._mfcc(inputs)
        if self._verbose:
            ConsoleLogger.status('_mfcc output size: {}'.format(mfcc_inputs.size()))

        normalized_mfcc_inputs = self.batch_norm(mfcc_inputs)
        # print("normalized_mfcc_inputs.size()", normalized_mfcc_inputs.size())
        # print("normalized_mfcc_inputs[0, :, 0]", normalized_mfcc_inputs[0, :, 0])

        x_conv_1 = F.relu(self._conv_1(normalized_mfcc_inputs))
        if self._verbose:
            ConsoleLogger.status('x_conv_1 output size: {}'.format(x_conv_1.size()))

        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self._verbose:
            ConsoleLogger.status('_conv_2 output size: {}'.format(x.size()))
        
        x_conv_3 = F.relu(self._conv_3(x))
        if self._verbose:
            ConsoleLogger.status('_conv_3 output size: {}'.format(x_conv_3.size()))

        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        if self._verbose:
            ConsoleLogger.status('_conv_4 output size: {}'.format(x_conv_4.size()))

        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self._verbose:
            ConsoleLogger.status('x_conv_5 output size: {}'.format(x_conv_5.size()))

        x_residual_stack = self._residual_stack(x_conv_5)
        if self._verbose:
            ConsoleLogger.status('_residual_stack output size: {}'.format(x_residual_stack.size()))

        x = self._conv_out(x_residual_stack)
        if self._verbose:
            ConsoleLogger.status('_conv_out output size: {}'.format(x.size()))

        return x
