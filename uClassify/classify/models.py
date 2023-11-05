import os
import shutil
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models
from classify.custom_user_manager import CustomUserManager

class CustomUser(AbstractUser):
  email = models.EmailField(unique=True)
  date_joined = models.DateTimeField(auto_now_add=True)
  is_active = models.BooleanField(default=True)
  is_staff = models.BooleanField(default=False)
  
  objects = CustomUserManager()

  # USERNAME_FIELD = 'email'
  REQUIRED_FIELDS = []

  def __str__(self):
    return self.email

class CustomizedImageClassificationModel(models.Model):
  VISION_TRANSFORMER = 'ViT'
  INCEPTION_V3 = 'IV3'
  RESNET_V2_50 = 'RN2'
  MOBILENET_V2 = 'MN2'
  MOBILENET_V3 = 'MN3'
  CLASSIFICATION_MODEL_CHOICES = [
    (VISION_TRANSFORMER, 'Vision Transformer'),
    (INCEPTION_V3, 'Inception V3'),
    (RESNET_V2_50, 'ResNet V2 50'),
    (MOBILENET_V2, 'MobileNet V2'),
    (MOBILENET_V3, 'MobileNet V3'),
  ]
  TENSORFLOW_HUB_URLS = {
    INCEPTION_V3: "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
    RESNET_V2_50: "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
    MOBILENET_V2: "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
    MOBILENET_V3: "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/feature_vector/5"
  }

  owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
  def __file_path(self, filename):
    owner_id = self.owner.id
    if not settings.PRODUCTION:
      return os.path.join(settings.MEDIA_ROOT, 'customized_models', str(owner_id), filename)
    else:
      return os.path.join('customized_models', str(owner_id), filename)

  model_type = models.CharField(
    max_length=3,
    choices=CLASSIFICATION_MODEL_CHOICES,
    default=INCEPTION_V3
  )
  def path_to_dataset(self):
    return self.__file_path('dataset')

  def path_to_model(self):
    if self.model_type == self.VISION_TRANSFORMER:
      return self.__file_path('model')
    else:
      return self.__file_path('model.keras')
  model_path = models.FileField(null=True)
  dataset_path = models.FileField(null=True)

  labels_list = models.JSONField(null=True, default=list)

  def delete(self, *args, **kwargs):
    if settings.PRODUCTION:
      if self.dataset_path:
        storage, name = self.dataset_path.storage, self.dataset_path.name
        storage.delete(name)
      if self.model_path:
        storage, name = self.model_path.storage, self.model_path.name
        storage.delete(name)
    else:
      if self.dataset_path:
        storage, path = self.dataset_path.storage, self.dataset_path.path
        storage.delete(path)
      if self.model_path:
        storage, path = self.model_path.storage, self.model_path.path
        if self.model_type == CustomizedImageClassificationModel.VISION_TRANSFORMER:
          shutil.rmtree(path)
        else:
          storage.delete(path)
    super().delete(*args, **kwargs)

class TrainingModelTask(models.Model):
  owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
  model = models.ForeignKey(CustomizedImageClassificationModel, on_delete=models.CASCADE)
  task_id = models.TextField(null=False, primary_key=True)

class EmailVerification(models.Model):
  user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
  verification_code = models.CharField(max_length=6)
  is_verified = models.BooleanField(default=False)
