import evaluate
import fire
import numpy as np
import os
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def main(ds_dir, outdir, model_name, num_labels=10):

    # labeled alpaca dataset with 10 bins
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_json_path = os.path.join(ds_dir, 'train.json')
    test_json_path = os.path.join(ds_dir, 'test.json')

    train_set = load_dataset('json', data_files=train_json_path)['train']
    test_set = load_dataset('json', data_files=test_json_path)['train']

    def tokenize_function(sequence):
        return tokenizer(sequence['prompt'], padding='max_length', truncation=True)

    encoded_train = train_set.map(tokenize_function, batched=True)
    encoded_test = test_set.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
    metric = evaluate.load('mse')

    def compute_metric(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    os.system("mkdir -p model/cache")
    args = TrainingArguments(
        os.path.join('model', 'cache', model_name),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='mse',
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_train,
        eval_dataset=encoded_test,
        compute_metrics=compute_metric,
    )

    trainer.train()
    os.system(f'mkdir -p {outdir}')
    model.save_pretrained(os.path.join('model', outdir, model_name))

    os.system('rm -rf wandb')


if __name__ == '__main__':
    fire.Fire(main)
