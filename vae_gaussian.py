import torch
from torch import nn
import torch.nn.functional as F
from wavenet_model import *

class GaussianVAE(nn.Module):
    def __init__(self, encoder, decoder, scale=1.):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.scale = scale

    def encode(self, x):
        '''
        x: (n_batches, length) with dtype [output_dim]
        return: tuple((n_batches, length); mu, log_variance)
        '''
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        '''
        z: (n_batches, latent_dim) with dtype float 
        return: (n_batches, output_dim, length) with dtype one_hot over output_dim 
        '''
        return self.decoder(z)

    def forward(self, x):
        '''
        x: as expected by `encode` 
        return: (p_x: (n_batches, output_dim, length) with dtype logits over output_dim 
                 mu, logvar: tuple((n_batches, length); mu, log_variance) 
        '''
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        p_x = self.decode(z)
        return p_x, mu, logvar
    
    def loss(self, p_x, x, mu, logvar):
        '''
        p_x: (n_batches, output_dim, length) with dtype logits over output_dim
        x: (n_batches, length) with dtype [output_dim]
        mu, logvar: tuple((n_batches, length); mu, log_variance) 
        return: loss with dtype float
        '''
        cross_entropy = F.cross_entropy(p_x, x, reduction='sum') / x.size(0)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return cross_entropy + kl_divergence

class GaussianWaveNetEncoder(nn.Module):
    def __init__(self, wavenet_args, use_continuous_one_hot=True):
        super().__init__()
        self.use_continuous_one_hot = use_continuous_one_hot
        if self.use_continuous_one_hot:
            assert wavenet_args["out_classes"] == 2
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field

    def forward(self, input):
        padded_input = F.pad(input, (self.padding_left, 0))
        one_hot_padded_input = torch.zeros(padded_input.size(0), self.wavenet.in_classes, padded_input.size(1))
        one_hot_padded_input.scatter_(1, padded_input.unsqueeze(1), 1.)
        padded_output = self.wavenet.wavenet(one_hot_padded_input, self.wavenet.wavenet_dilate)
        output = padded_output[:, :, -input.size(-1):]
        if self.use_continuous_one_hot:
            mu = output[:, 0, :]
            logvar = output[:, 1, :]
        else:
            mu = output[:, :self.wavenet.out_classes//2, :]
            logvar = output[:, self.wavenet.out_classes//2:, :]
        return mu, logvar

class ContinuousToOneHot(nn.Module):
    def __init__(self, n_classes=256):
        super().__init__()
        self.n_classes = n_classes
    def forward(self, x):
        '''
        x: (n_batches, length), dtype: float
        returns: (n_batches, n_classes, length), dtype: float
        '''
        x += .5
        dist = torch.distributions.normal.Normal(x.unsqueeze(-1), 0.05)
        positions = torch.arange(0, self.n_classes).float() / self.n_classes
        positions_shifted = positions + 1/self.n_classes
        cdfs = dist.cdf(positions.unsqueeze(0).unsqueeze(0))
        cdfs_shifted = dist.cdf(positions_shifted.unsqueeze(0).unsqueeze(0))
        cdfs[:, :, 0] = 0
        cdfs_shifted[:, :, -1] = 1
        continuous_one_hot = cdfs_shifted - cdfs
        return continuous_one_hot.transpose(1, 2)

class GaussianWaveNetDecoder(nn.Module):
    def __init__(self, wavenet_args, use_continuous_one_hot=True):
        super().__init__()
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field
        self.use_continuous_one_hot = use_continuous_one_hot
        if self.use_continuous_one_hot:
            self.continuous_to_one_hot = ContinuousToOneHot(wavenet_args["in_classes"])

    def forward(self, input):
        padded_input = F.pad(input, (self.padding_left, 0))
        if self.use_continuous_one_hot:
            one_hot_padded_input = self.continuous_to_one_hot(padded_input)
        else:
            one_hot_padded_input = padded_input
        padded_output = self.wavenet.wavenet(one_hot_padded_input, self.wavenet.wavenet_dilate)
        output = padded_output[:, :, -input.size(-1):]
        return output