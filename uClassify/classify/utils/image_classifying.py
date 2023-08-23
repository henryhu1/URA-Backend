from PIL import Image
from transformers import ViTForImageClassification
from classify.static_image_classifier import StaticImageClassifier

def classify_image(image_file, classifier_path=None):
  if not classifier_path:
    classifier_model = StaticImageClassifier.get_model()
  else:
    classifier_model = ViTForImageClassification.from_pretrained(classifier_path)
  tokenizer = StaticImageClassifier.get_tokenizer()

  image = Image.open(image_file)
  image = image.resize((224, 224))
  input = tokenizer(image, return_tensors="pt")
  output = classifier_model(**input)
  logits = output.logits
  prediction = logits.argmax(1).item()

  return classifier_model.config.id2label[prediction]