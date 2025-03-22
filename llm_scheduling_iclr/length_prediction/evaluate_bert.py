import fire
import json
import numpy as np
import os
import time
import torch
import tqdm
from collections import Counter
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main(model_path, num_labels, test_data_path, output_path):

    input_column_name = 'prompt'
    output_column_name = 'label'
    test_data = load_dataset('json', data_files=test_data_path)['train']


    # evaluate on test data
    predictions = list()
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels).to('cuda')
    model.eval()
    ids = test_data['id']
    texts = test_data[input_column_name]
    pbar = tqdm.tqdm(total=len(texts))
    with torch.no_grad():
        for id, seq in zip(ids, texts):
            query = tokenizer(seq, truncation=True, padding='max_length', return_tensors='pt').to('cuda')
            logits = model(**query).logits                                                                                                                                                                                                                                                                                       
            logits_np = logits[0].detach().cpu().numpy().tolist()

            predictions.append({
                'id': id,
                'logits': logits_np
            })

            pbar.update(1)

    pbar.close()

    with open(output_path, 'w') as fout:
        json.dump(predictions, fout, indent=2)


if __name__ == '__main__':
    fire.Fire(main)
