import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import time
import argparse
import json
import math
import os

from wavenet_vocoder import WaveNet
from audio_data import AudioDataset

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from wavenet_utils import parameter_count

from apex import amp

dtype = torch.FloatTensor
ltype = torch.LongTensor

def load_checkpoint(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    step = checkpoint_dict['step']
    # optimizer.load_state_dict()
    model.load_state_dict(checkpoint_dict['model'])
    print("Loaded checkpoint '{}' (step {})" .format(checkpoint_path, step))
    return model, checkpoint_dict['optimizer'], step

def save_checkpoint(model, optimizer, step, filepath):
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
weight_decay  = train_args["weight_decay"]
continue_training_at_step = train_args["continue_training_at_step"]
snapshot_name = os.path.splitext(os.path.basename(args.config))[0]
snapshot_path = f"snapshots/{snapshot_name}"
snapshot_interval = train_args["snapshot_interval"]
lr = train_args["lr"]
device_name = config["device"]
dataset_path = config["dataset_path"]
load_path = config["load_path"]
type = config.get("type", "wavenet")
use_apex = config.get("use_apex", False)

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)
    print(f"creating path {snapshot_path}")

assert type == "wavenet"
class WaveNetWrapper(WaveNet):
    def __init__(self, *args, **kwargs):
        print("kwargs", kwargs)
        super().__init__(*args, **kwargs)

    def forward(self, input):
        one_hot_input = torch.zeros(input.size(0), self.out_channels, input.size(1))
        one_hot_input.scatter_(1, input.unsqueeze(1), 1.)
        return super().forward(one_hot_input)
model = WaveNetWrapper(**wavenet_args)

dataset = AudioDataset(dataset_path, model.receptive_field*8)        

print('the dataset has ' + str(len(dataset)) + ' items')
print(f'each item has length {dataset.len_sample}')
print(f'The WaveNet has {parameter_count(model)} parameters')

print('start training...')
clip = None

pin_memory = False
device = torch.device(device_name)
if not device_name == "cpu":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = model.to(device)
    pin_memory = True

optimizer=optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
if "optimizer_state_dict" in globals():
    optimizer.load_state_dict(optimizer_state_dict)

### APEX MIXED-PRECISION
if use_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = nn.DataParallel(model)
if not load_path == "":
    (_, optimizer_state_dict, continue_training_at_step) = load_checkpoint(load_path, model)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=pin_memory,
)

writer = SummaryWriter(comment="_"+snapshot_name)
step = continue_training_at_step
model.train()


print("epochs: ", epochs)
for current_epoch in range(epochs):
    print("\nepoch\n", current_epoch)
    tic = time.time()
    for x in iter(dataloader):
        x = x.long().to(device)
        
        with torch.autograd.profiler.profile(enabled=False, use_cuda=True) as prof:
            # x.shape torch.Size([2, 1, 8000])
            # y.shape torch.Size([2, 8000, 1])
            # y_hat.shape torch.Size([2, 30, 8000])
            '''
            p_x: (n_batches, output_dim, length) with dtype logits over output_dim
            x: (n_batches, length) with dtype [output_dim]
            q_z: (n_batches, categorical_dim, latent_dim) with dtype logits over categorical_dim)
            return: loss with dtype float
            '''
            y = x[:,  1:]
            y_hat = model(x)[:, :, :-1]
            loss = F.cross_entropy(y_hat, y, reduction='sum') / x.size(0)

            optimizer.zero_grad()
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            loss = loss.item()
            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            step += 1
        if prof is not None:
            prof.export_chrome_trace("chrome_trace")

        if step == 100:
            toc = time.time()
            print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

        if step % snapshot_interval == 0:
            if snapshot_path is None:
                continue
            path = snapshot_path + '/' + snapshot_name + '_' + str(step)
            save_checkpoint(model, optimizer, step, path)
        if step % 10 == 0:
            bits_per_dimension = (loss/math.log(2)) / x.size(-1)
            writer.add_scalar('Loss/train', loss, global_step=step)
            writer.add_scalar('bits/dimension', bits_per_dimension, global_step=step)
            print(f"\rstep {step}, loss {loss}".ljust(70), end='\r')

print("finished")
