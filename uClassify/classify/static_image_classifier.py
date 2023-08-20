from transformers import ViTForImageClassification, ViTImageProcessor
from django.conf import settings
import os

class StaticImageClassifier:
  # Check if STATIC_ROOT is set (common in production). If not, use STATICFILES_DIRS
  static_location = settings.STATIC_ROOT if hasattr(settings, "STATIC_ROOT") else settings.STATICFILES_DIRS[0]
  model_path = os.path.join(static_location, "vit-base-patch16-224")
  tokenizer_path = os.path.join(static_location, "vit-base-patch16-224-in21k")

  @classmethod
  def get_model(self):
    if not os.path.exists(self.model_path):
      model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", cache_dir=settings.MODEL_CACHE_DIR)
      model.save_pretrained(self.model_path)
    else:
      model = ViTForImageClassification.from_pretrained(self.model_path)
    return model

  @classmethod
  def get_tokenizer(self):
    if not os.path.exists(self.tokenizer_path):
      tokenizer = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", cache_dir=settings.MODEL_CACHE_DIR)
      tokenizer.save_pretrained(self.tokenizer_path)
    else:
      tokenizer = ViTImageProcessor.from_pretrained(self.tokenizer_path)
    return tokenizer
