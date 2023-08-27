import torch
import numpy as np
from datasets import load_dataset, load_metric
from tensorflow import keras, data
from tensorflow_hub import KerasLayer
from transformers import ViTForImageClassification, TrainingArguments, Trainer
from classify.static_image_classifier import StaticImageClassifier

TOKENIZER = StaticImageClassifier.get_tokenizer()
BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
NORMALIZATION = keras.layers.Rescaling(1./255)
AUTOTUNE = data.AUTOTUNE

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

def get_training_and_validation_datasets(path_to_dataset, dataset_split):
  ds = keras.utils.image_dataset_from_directory(
    path_to_dataset,
    validation_split=dataset_split,
    subset="both",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
  )
  #TODO normalization util
  ds[0] = ds[0].map(lambda x, y: (NORMALIZATION(x), y))
  ds[1] = ds[1].map(lambda x, y: (NORMALIZATION(x), y))
  # train_ds = train_ds.cache().prefetech(buffer_size=AUTOTUNE)
  # val_ds = val_ds.cache().prefetech(buffer_size=AUTOTUNE)
  return ds

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

def get_hub_model(tensorflow_hub_url, num_classes):
  feature_extractor_layer = KerasLayer(
    tensorflow_hub_url,
    input_shape=(224, 224, 3),
    trainable=False
  )
  model = keras.Sequential([
    feature_extractor_layer,
    keras.layers.Dense(num_classes)
  ])
  model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
  )

  return model
