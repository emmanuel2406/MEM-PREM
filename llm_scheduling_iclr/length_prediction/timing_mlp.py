import json
import itertools
import numpy as np
import time
import torch
from torch import nn, optim
from tqdm import trange

N_TRIALS = 20
DEVICES = ['cpu', 'cuda']
BATCH_SIZES = [512, 1024, 2048]

with torch.no_grad():

    measurements = list()
    for device, bsize in itertools.product(DEVICES, BATCH_SIZES):

        # load model and test tensors
        X_test = torch.load('layer11_mlp/X_test.pt')
        Y_test = torch.load('layer11_mlp/Y_test.pt')

        num_samples = X_test.size(0)

        log = {
            "batch_size": bsize,
            "device": device
        }

        time_per_sample = list()
        per_trial_measurements = list()

        for trial in trange(N_TRIALS, desc=f"device {device} batch size {bsize}"):

            num_labels = 10
            model = nn.Sequential(
                nn.Linear(X_test.shape[1], 512),
                nn.ReLU(),
                nn.Linear(512, num_labels)
            )
            model.load_state_dict(torch.load('layer11_mlp/model.pth'))

            model.to(device)
            X_test, Y_test = X_test.to(device), Y_test.to(device)
            model.eval()

            # iterate through test set with batch size
            batch_times = list()
            for batch_idx in range(0, len(X_test), bsize):
                X_batch = X_test[batch_idx:batch_idx + bsize]

                if device == 'cpu':
                    start_time = time.time()
                    _ = torch.argmax(model(X_batch), dim=1)
                    batch_time = time.time() - start_time
                
                else:  # cuda
                    start_time = time.time()
                    _ = torch.argmax(model(X_batch), dim=1)
                    torch.cuda.synchronize()
                    batch_time = time.time() - start_time
                
                batch_times.append(batch_time)
            
            per_trial_measurements.append({
                "trial": trial,
                "per_batch_time": batch_times
            })
            
            time_per_sample.append(np.sum(batch_times) / num_samples)
        
        log['per_sample_time'] = time_per_sample
        log['per_batch_time_ntrials'] = per_trial_measurements
        measurements.append(log)

    with open('measurements/mlp_inference_time_raw.json', 'w') as fout:
        json.dump(measurements, fout)
