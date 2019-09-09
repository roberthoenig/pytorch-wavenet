import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import time
import argparse
import json

from wavenet_model import *
from wavenet_modules import *
from audio_data import WavenetDataset
from vae import VAE

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

dtype = torch.FloatTensor
ltype = torch.LongTensor

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    step = checkpoint_dict['step']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model.load_state_dict(checkpoint_dict['model'])
    print("Loaded checkpoint '{}' (step {})" .format(checkpoint_path, step))
    return model, optimizer, step

def save_checkpoint(model, optimizer, step, filepath):
    print("Saving model and optimizer state at step {} to {}".format(step, filepath))
    torch.save({'model': model.state_dict(),
                'step': step,
                'optimizer': optimizer.state_dict(),
                }, filepath)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str,
                    help='JSON file for configuration', required=True)
args = parser.parse_args()
with open(args.config) as f:
    data = f.read()
config = json.loads(data)
wavenet_args = config["wavenet_args"]
train_args = config["train_args"]
batch_size = train_args["batch_size"]
epochs = train_args["epochs"]
continue_training_at_step = train_args["continue_training_at_step"]
snapshot_name = train_args["snapshot_name"]
snapshot_interval = train_args["snapshot_interval"]
snapshot_path = train_args["snapshot_path"]
weight_decay  = train_args["weight_decay"]
lr = train_args["lr"]
gumbel_softmax_temperature  = train_args["gumbel_softmax_temperature"]
device_name = config["device"]
gpu_index = config["gpu_index"]

class WaveNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field

    def forward(self, input):
        padded_input = F.pad(input, (self.padding_left, 0))
        one_hot_padded_input = torch.zeros(padded_input.size(0), self.wavenet.classes, padded_input.size(1))
        one_hot_padded_input.scatter_(1, padded_input.unsqueeze(1), 1.)
        padded_output = self.wavenet.wavenet(one_hot_padded_input, self.wavenet.wavenet_dilate)
        output = padded_output[:, :, -input.size(-1):]
        return output

class WaveNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavenet = WaveNetModel(**wavenet_args)
        self.padding_left = self.wavenet.receptive_field

    def forward(self, input):
        padded_input = F.pad(input, (self.padding_left, 0))
        padded_output = self.wavenet.wavenet(padded_input, self.wavenet.wavenet_dilate)
        output = padded_output[:, :, -input.size(-1):]
        return output

model = VAE(
    WaveNetEncoder(),
    WaveNetDecoder(),
)
    
dataset = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                      item_length=model.encoder.wavenet.receptive_field*4,
                      target_length=model.encoder.wavenet.output_length,
                      file_location='train_samples/bach_chaconne',
                      test_stride=500,
                      one_hot=False)
print('the dataset has ' + str(len(dataset)) + ' items')
print(f'each item has length {dataset._item_length}')
n_parameters = model.encoder.wavenet.parameter_count() + model.decoder.wavenet.parameter_count()
print(f'The WaveNetVAE has {n_parameters} parameters')

print('start training...')
optimizer=optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
clip = None


load_path = ""
if not load_path == "":
    (_, _, continue_training_at_step) = load_checkpoint(load_path, model, optimizer)

pin_memory = False
device = torch.device(device_name)
if not device_name == "cpu":
    torch.cuda.set_device(gpu_index)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = model.to(device)
    pin_memory = True

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=pin_memory,
)

writer = SummaryWriter()
step = continue_training_at_step
model.train()
for current_epoch in range(epochs):
    print("epoch", current_epoch)
    tic = time.time()
    for (x, target) in iter(dataloader):
        x = x.to(device)
        p_x, q_z = model(x, gumbel_softmax_temperature)
        loss = model.loss(p_x, x, q_z)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if clip is not None:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        step += 1

        if step == 100:
            toc = time.time()
            print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

        if step % snapshot_interval == 0:
            if snapshot_path is None:
                continue
            print("taking a snapshot")
            path = snapshot_path + '/' + snapshot_name + '_' + str(step)
            save_checkpoint(model, optimizer, step, path)
        writer.add_scalar('Loss/train', loss, global_step=step)
        print(f"step {step}, loss {loss}")
