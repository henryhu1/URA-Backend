from PIL import Image
import numpy as np
import os
from tensorflow import keras, math
from tensorflow_hub import KerasLayer
from transformers import ViTForImageClassification
from classify.static_image_classifier import StaticImageClassifier

def classify_image(image_file, is_vision_transformer=True, classifier_path=None, labels_list=[]):
  image = Image.open(image_file)
  image = image.resize((224, 224))
  if not is_vision_transformer:
    classifier_model = keras.models.load_model(classifier_path, custom_objects={'KerasLayer': KerasLayer})
    image_array = np.array(image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    predictions = classifier_model.predict(image_batch)
    predicted_id = math.argmax(predictions, axis=-1)
    predicted_label = labels_list[predicted_id.numpy()[0]]
    return predicted_label
  else:
    if not classifier_path:
      classifier_model = StaticImageClassifier.get_model()
    else:
      classifier_model = ViTForImageClassification.from_pretrained(classifier_path)
    tokenizer = StaticImageClassifier.get_tokenizer()

    input = tokenizer(image, return_tensors="pt")
    output = classifier_model(**input)
    logits = output.logits
    prediction = logits.argmax(1).item()

    return classifier_model.config.id2label[prediction]