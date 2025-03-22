`per_layer_classification.py`: train MLP models with per-layer embeddings. The input can be averaged prompt tokens, or intermediate steps.

`finetune_bert.py` and `evaluate_bert.py`: finetune pre-trained distilBERT model for output length prediction, and evaluate on test set.

`timing_bert.py` and `timing_mlp.py`: record the inference time of finetuned distilBERT model and MLP model, under various batch size and device settings.