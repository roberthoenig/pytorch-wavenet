import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.transforms import *
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution

from wavenet_model import *


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature, hard=False):
    """
    input: (n_batches, categorical_dim, *) with dtype logits
    return: [n_batches, categorical_dim, *] with dtype almost-one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind.unsqueeze(1), 1)
    y_hard = (y_hard - y).detach() + y
    return y_hard

class VAE(nn.Module):
    def __init__(self, encoder, decoder, hard_gumbel_softmax=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hard_gumbel_softmax = hard_gumbel_softmax

    def p_x_to_sample(self, p_x):
        '''
        p_x: (n_batches, output_dim, length)
        '''
        m = Categorical(logits=p_x.permute(0, 2, 1))
        return m.sample().float() / (output_dim - 1)

    def encode(self, x):
        '''
        x: (n_batches, length) with dtype [output_dim]
        return: (n_batches, categorical_dim, latent_dim) with dtype logits
        '''
        return self.encoder(x)

    def decode(self, z):
        '''
        z: (n_batches, categorical_dim, latent_dim) with dtype one_hot over output_dim 
        return: (n_batches, output_dim, length) with dtype one_hot over output_dim 
        '''
        return self.decoder(z)

    def forward(self, x, temperature):
        '''
        x: as expected by `encode` 
        return: (p_x: (n_batches, output_dim, length) with dtype logits over output_dim 
                 q_z: (n_batches, categorical_dim, latent_dim) with dtype logits over categorical_dim)  
        '''
        q_z = self.encode(x)
        z = gumbel_softmax(q_z, temperature, self.hard_gumbel_softmax)
        p_x = self.decode(z)
        return p_x, q_z
    
    def loss(self, p_x, x, q_z, posterior_entropy_penalty_coeff=0.0):
        '''
        p_x: (n_batches, output_dim, length) with dtype logits over output_dim
        x: (n_batches, length) with dtype [output_dim]
        q_z: (n_batches, categorical_dim, latent_dim) with dtype logits over categorical_dim)
        return: loss with dtype float
        '''
        cross_entropy = F.cross_entropy(p_x, x, reduction='sum') / x.size(0)
        log_ratio = F.log_softmax(q_z, dim=1) + torch.log(torch.tensor(q_z.size(1)).float())
        kl_divergence = torch.sum(F.softmax(q_z, dim=1) * log_ratio) / x.size(0)
        posterior_entropy_penalty = Categorical(logits=q_z.transpose(1 ,2)).entropy().sum() / x.size(0)
        
        return {
            'loss': cross_entropy + kl_divergence + posterior_entropy_penalty_coeff * posterior_entropy_penalty,
            'cross_entropy': cross_entropy,
            'kl_divergence': kl_divergence,
            'posterior_entropy_penalty': posterior_entropy_penalty
        }

class WaveNetEncoder(nn.Module):
    def __init__(self, wavenet_args):
        super().__init__()
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field

    def forward(self, input):
        padded_input = F.pad(input, (self.padding_left, 0))
        one_hot_padded_input = torch.zeros(padded_input.size(0), self.wavenet.in_classes, padded_input.size(1))
        one_hot_padded_input.scatter_(1, padded_input.unsqueeze(1), 1.)
        padded_output = self.wavenet.wavenet(one_hot_padded_input, self.wavenet.wavenet_dilate)
        output = padded_output[:, :, -input.size(-1):]
        return output

class WaveNetDecoder(nn.Module):
    def __init__(self, wavenet_args, flip_decoder=True):
        super().__init__()
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field
        self.is_input_one_hot = True
        self.flip_decoder = flip_decoder

    def forward(self, input):
        if self.flip_decoder:
            input = input.flip(-1)
        padded_input = F.pad(input, (self.padding_left, 0))
        if self.is_input_one_hot:
            padded_output = self.wavenet.wavenet(padded_input, self.wavenet.wavenet_dilate)
        else:
            one_hot_padded_input = torch.zeros(padded_input.size(0), self.wavenet.in_classes, padded_input.size(1))
            one_hot_padded_input.scatter_(1, padded_input.unsqueeze(1), 1.)
            padded_output = self.wavenet.wavenet(one_hot_padded_input, self.wavenet.wavenet_dilate)
            
        output = padded_output[:, :, -input.size(-1):]
        if self.flip_decoder:
            output = output.flip(-1)
        return output

class LogisticMixtureDistribution():
    # all: (n_batches, n_latents, n_modes)
    def __init__(self, loc, scale, categorical_logits):
        self.loc = loc.unsqueeze(-2)
        self.scale = scale.unsqueeze(-2)
        self.categorical_logits = categorical_logits.unsqueeze(-2)
    # x: (n_batches, n_latents, n_xs)
    # cdfs: (n_batches, n_latents, n_xs, n_modes)
    # returns: (n_batchs, n_latents, n_xs)
    def cdf(self, x):    
        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv, AffineTransform(loc=self.loc, scale=self.scale, event_dim=1)]
        logistic = TransformedDistribution(base_distribution, transforms)
        x = x.unsqueeze(-1).expand(-1, -1, -1, self.loc.size(-1))
        cdfs = logistic.cdf(x)
        return torch.sum(F.softmax(self.categorical_logits, dim=-1) * cdfs, dim=-1)

class MultimodalWaveNetEncoder(nn.Module):
    def __init__(self, wavenet_args):
        super().__init__()
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field

    def forward(self, input):
        padded_input = F.pad(input, (self.padding_left, 0))
        one_hot_padded_input = torch.zeros(padded_input.size(0), self.wavenet.in_classes, padded_input.size(1))
        one_hot_padded_input.scatter_(1, padded_input.unsqueeze(1), 1.)
        padded_output = self.wavenet.wavenet(one_hot_padded_input, self.wavenet.wavenet_dilate)
        output = padded_output[:, :, -input.size(-1):].transpose(-1, -2)
        
        loc, scale, categorical_logits = output.chunk(3, dim=-1)
        loc = F.sigmoid(loc)
        scale = F.sigmoid(scale) * .05
        dist = LogisticMixtureDistribution(loc, scale, categorical_logits)

        n_classes = 256 # this value must match the decoder's in_classes
        positions = torch.arange(0, n_classes).float() / n_classes
        positions_shifted = positions + 1/n_classes
        cdfs = dist.cdf(positions.unsqueeze(0).unsqueeze(0))
        cdfs_shifted = dist.cdf(positions_shifted.unsqueeze(0).unsqueeze(0))
        quantized_dist = cdfs_shifted - cdfs
        quantized_dist /=  quantized_dist.sum(-1).unsqueeze(-1)
        quantized_dist += 1e-12
        torch.save((quantized_dist, input, loc, scale, categorical_logits), "arrays.pt")
        return quantized_dist.transpose(-1, -2).log()
