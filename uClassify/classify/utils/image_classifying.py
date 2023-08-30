from django.conf import settings
from PIL import Image
import boto3
import numpy as np
import os
import shutil
import tempfile
from tensorflow import keras, math
from tensorflow_hub import KerasLayer
from transformers import ViTForImageClassification
from classify.static_image_classifier import StaticImageClassifier

def classify_image(image_file, is_vision_transformer=True, classifier_path=None, labels_list=[]):
  image = Image.open(image_file)
  image = image.resize((224, 224))
  if not is_vision_transformer:
    if settings.PRODUCTION:
      temp_model_dir = tempfile.mkdtemp()
      temp_model_path = os.path.join(temp_model_dir, 'temp.keras')
      s3 = boto3.client('s3')
      bucket_name = settings.AWS_STORAGE_BUCKET_NAME
      file_key = settings.AWS_LOCATION + '/' + classifier_path
      s3.download_file(bucket_name, file_key, temp_model_path)
      classifier_model = keras.models.load_model(temp_model_path, custom_objects={'KerasLayer': KerasLayer})
      shutil.rmtree(temp_model_dir)
    else:
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