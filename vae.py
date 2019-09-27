import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.transforms import *
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution

from convolutional_encoder import ConvolutionalEncoder

from wavenet_model import *

from wavenet_vocoder import WaveNet

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

    def decode(self, z, x):
        '''
        z: (n_batches, categorical_dim, latent_dim) with dtype one_hot over output_dim 
        return: (n_batches, output_dim, length) with dtype one_hot over output_dim 
        '''
        return self.decoder(z, x)

    def forward(self, x, temperature):
        '''
        x: as expected by `encode` 
        return: (p_x: (n_batches, output_dim, length) with dtype logits over output_dim 
                 q_z: (n_batches, categorical_dim, latent_dim) with dtype logits over categorical_dim)  
        '''
        q_z = self.encode(x)
        z = gumbel_softmax(q_z, temperature, self.hard_gumbel_softmax)
        if type(self.decoder) == BetterWaveNetDecoder:
            p_x = self.decode(z, x)
        else:
            p_x = self.decode(z)
        return p_x, q_z
    
    def kl_divergence_ar_prior(self, q_z, a):
        '''
        q_z: (n_batches, n_categorical_choices, n_latents) with dtype logits over n_categorical_choices)
        '''
        N = torch.tensor(q_z.size(1)).float()
        a = torch.tensor(a)
        log_q_z_probs = F.log_softmax(q_z, dim=1)
        q_z_probs = F.softmax(q_z, dim=1)
        kl_z1 = torch.sum(q_z_probs[:, :, 0] * (log_q_z_probs[:, :, 0] + torch.log(N)))
        kl_zRest = (
            torch.sum(q_z_probs[:, :, 1:] * (log_q_z_probs[:, :, 1:] + torch.log(N-1) - torch.log(1-a))) +
            torch.sum(q_z_probs[:, :, 1:] * q_z_probs[:, :, :-1] * (torch.log(1-a) - (torch.log(N-1) + torch.log(a))))
        )
        return (kl_z1 + kl_zRest) / q_z.size(0)

    def kl_divergence_uniform_prior(self, q_z):
        '''
        q_z: (n_batches, n_categorical_choices, n_latents) with dtype logits over n_categorical_choices)
        '''
        log_ratio = F.log_softmax(q_z, dim=1) + torch.log(torch.tensor(q_z.size(1)).float())
        return torch.sum(F.softmax(q_z, dim=1) * log_ratio) / q_z.size(0)

    def loss(self, p_x, x, q_z, posterior_entropy_penalty_coeff=0.0, ar_factor=None):
        '''
        p_x: (n_batches, output_dim, length) with dtype logits over output_dim
        x: (n_batches, length) with dtype [output_dim]
        q_z: (n_batches, categorical_dim, latent_dim) with dtype logits over categorical_dim)
        return: loss with dtype float
        '''
        # print("p_x.size()", p_x.size())
        # print("x.size()", x.size())
        # print("x.size()", x.size())
        # print("q_z.size()", q_z.size())
        cross_entropy = F.cross_entropy(p_x, x, reduction='sum') / x.size(0)
        kl_divergence = self.kl_divergence_uniform_prior(q_z) if ar_factor is None else self.kl_divergence_ar_prior(q_z, ar_factor)
        posterior_entropy_penalty = Categorical(logits=q_z.transpose(1 ,2)).entropy().sum() / q_z.size(0)
        
        return {
            'loss': cross_entropy + kl_divergence + posterior_entropy_penalty_coeff * posterior_entropy_penalty,
            'cross_entropy': cross_entropy,
            'kl_divergence': kl_divergence,
            'posterior_entropy_penalty': posterior_entropy_penalty
        }

class BetterWaveNetDecoder(nn.Module):
    def __init__(self, wavenet_args):
        super().__init__()
        self.wavenet = WaveNet(**wavenet_args)

    def forward(self, one_hot_z, x):
        output = self.wavenet.forward(x=self.one_hot(x), c=one_hot_z)
        p_x = torch.cat([torch.ones(output.size(0), output.size(1), 1), output[:, :, :-1]], dim=-1)
        return p_x

    def one_hot(self, input):
        one_hot_input = torch.zeros(input.size(0), self.wavenet.out_channels, input.size(1), device=input.device)
        one_hot_input.scatter_(1, input.unsqueeze(1), 1.)
        return one_hot_input

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

class OneHotConvolutionalEncoder(ConvolutionalEncoder):
    def __init__(self, wavenet_args):
        super().__init__(**wavenet_args)
        self.in_classes = wavenet_args["features_filters"]

    def forward(self, input):
        one_hot_input = torch.zeros(input.size(0), self.in_classes, input.size(1))
        one_hot_input.scatter_(1, input.unsqueeze(1), 1.)
        return super().forward(one_hot_input)