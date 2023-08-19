from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
import torch
import numpy as np
from datasets import load_metric

class ImageClassifier:
  def __init__(self, labels=None):
    if (labels == None):
      self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    else:
      self.model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
      )
    self.labels = labels
    self.tokenizer = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

  def collate_fn(batch):
    return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['label'] for x in batch])
    }

  def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

  training_args = TrainingArguments(
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

  def getTrainer(self, dataset):
    return Trainer(
      model=self.model,
      args=self.training_args,
      data_collator=self.collate_fn,
      compute_metrics=self.compute_metrics,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      tokenizer=self.tokenizer,
    )

  def classify_image(self, image):
    input = self.tokenizer(image, return_tensors="pt")
    output = self.model(**input)
    logits = output.logits
    prediction = logits.argmax(1).item()
    print(prediction)
    return self.model.config.id2label[prediction]
