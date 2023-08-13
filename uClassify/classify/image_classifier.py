from transformers import ViTForImageClassification, ViTImageProcessor

class ImageClassifier:
  def __init__(self):
    self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    self.tokenizer = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

  def classify_image(self, image):
    input = self.tokenizer(image, return_tensors="pt")
    output = self.model(**input)
    logits = output.logits
    prediction = logits.argmax(1).item()
    print(prediction)
    return self.model.config.id2label[prediction]
