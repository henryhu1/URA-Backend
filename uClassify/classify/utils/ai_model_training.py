import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from classify.static_image_classifier import StaticImageClassifier

TOKENIZER = StaticImageClassifier.get_tokenizer()

def transform(example_batch):
  # Take a list of PIL images and turn them to pixel values
  inputs = TOKENIZER([x for x in example_batch['image']], return_tensors='pt')

  # Don't forget to include the labels!
  inputs['label'] = example_batch['label']
  return inputs

def collate_fn(batch):
  return {
    'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
    'labels': torch.tensor([x['label'] for x in batch])
  }

def compute_metrics(p):
  metric = load_metric("accuracy")
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def get_dataset(path_to_dataset, dataset_split):
  ds = load_dataset(path_to_dataset, split="train").train_test_split(test_size=dataset_split)
  return ds.with_transform(transform)

def get_pretrained_model(labels):
  return ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
  )

def get_training_arguments():
  #TODO let user input variables
  return TrainingArguments(
    output_dir="./vit_output",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
  )

def get_trainer(model, ds):
  return Trainer(
    model=model,
    args=get_training_arguments(),
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=TOKENIZER,
  )
