import json
import itertools
import numpy as np
import os
import time
import torch
import tqdm
from collections import Counter
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

N_TRIALS = 20
DEVICES = ['cuda', ]
BATCH_SIZES = [512, 1024]

BERT_MODEL_PATH = "bert/alpaca-bert"

with torch.no_grad():

    measurements = list()

    for device, bsize in itertools.product(DEVICES, BATCH_SIZES):

        log = {
            "batch_size": bsize,
            "device": device
        }

        # get tokenized prompts
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        test_data = load_dataset('json', data_files='bert/test.json')['train']
        texts = [x['prompt'] for x in test_data]
        num_samples = len(texts)
        encoded_texts = tokenizer(texts, truncation=True, padding='max_length', return_tensors='pt')

        # batching
        dataset = TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'])

        # load model
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_PATH,
            num_labels=10)
        model.eval()

        if device == 'cuda':
            model.to('cuda')
        
        tps = list()
        ptm = list()
        
        for trial in tqdm.trange(N_TRIALS, desc=f"device {device} batch size {bsize}"):

            per_batch_time = list()
            dataloader = DataLoader(dataset, batch_size=bsize)
            for batch in dataloader:
                if device == 'cuda':
                    input_ids_batch, attention_masks_batch = batch[0].cuda(), batch[1].cuda()
                else:
                    input_ids_batch, attention_masks_batch = batch

                if device == 'cuda':
                    t1 = time.time()
                    _ = model(input_ids_batch, attention_mask=attention_masks_batch)
                    torch.cuda.synchronize()
                    batch_time = time.time() - t1
                    per_batch_time.append(batch_time)
                
                else:
                    t1 = time.time()
                    _ = model(input_ids_batch, attention_mask=attention_masks_batch)
                    batch_time = time.time() - t1
                    per_batch_time.append(batch_time)

            tps.append(np.sum(per_batch_time) / num_samples)    
            ptm.append({
                "trial": trial,
                "per_batch_time": per_batch_time
            })
        
        log['per_sample_time'] = tps
        log['per_batch_time_ntrials'] = ptm
        measurements.append(log)

    with open('measurements/bert_inference_time_raw.json', 'w') as fout:
        json.dump(measurements, fout)
