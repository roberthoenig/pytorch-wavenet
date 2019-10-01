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


class UnstridedConvolutionalEncoder(nn.Module):
    
    def __init__(self, num_hiddens, num_outs, num_residual_layers, num_residual_hiddens,
        use_kaiming_normal, features_filters, verbose=False, use_dilation=False):

        super().__init__()
        print(f"ConvolutionalEncoder.__init__ called with num_hiddens = {num_hiddens}")

        dilation_base = 2 if use_dilation else 1
        self.kernel_size = 3

        self._conv_1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=self.kernel_size,
            use_kaiming_normal=use_kaiming_normal,
            dilation=dilation_base**0,
            pad_right_only=True
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=self.kernel_size,
            use_kaiming_normal=use_kaiming_normal,
            dilation = dilation_base**1,
            pad_right_only=True
        )

        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            dilation = dilation_base**2,
            pad_right_only=True
        )

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=self.kernel_size,
            use_kaiming_normal=use_kaiming_normal,
            dilation = dilation_base**3,
            pad_right_only=True
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            dilation = dilation_base**4,
            pad_right_only=True
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
            init_dilation = dilation_base**4,
            dilation_base = dilation_base,
            pad_right_only=True
        )

        """
        1 postprocessing convolution to obtain the desired num_outs
        channels.
        """
        self._conv_out = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_outs,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            dilation = dilation_base**4,
            pad_right_only=True
        )

        self._verbose = verbose

    def forward(self, inputs):
        if self._verbose:
            ConsoleLogger.status('inputs size: {}'.format(inputs.size()))

        x_conv_1 = F.relu(self._conv_1(inputs))
        if self._verbose:
            ConsoleLogger.status('x_conv_1 output size: {}'.format(x_conv_1.size()))

        x_conv_2 = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self._verbose:
            ConsoleLogger.status('_conv_2 output size: {}'.format(x_conv_2.size()))
        
        x_conv_3 = F.relu(self._conv_3(x_conv_2)) + x_conv_2
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
