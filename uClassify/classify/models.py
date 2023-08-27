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
  model_type = models.CharField(
    max_length=3,
    choices=CLASSIFICATION_MODEL_CHOICES,
    default=INCEPTION_V3
  )
  model_path = models.FileField(null=True)

  def path_to_dataset(self):
    return '{}/{}/{}'.format(settings.MEDIA_ROOT, self.owner.id, "dataset")

class TrainingModelTask(models.Model):
  owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
  model = models.ForeignKey(CustomizedImageClassificationModel, on_delete=models.CASCADE)
  task_id = models.TextField(null=False, primary_key=True)

class EmailVerification(models.Model):
  user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
  verification_code = models.CharField(max_length=6)
  is_verified = models.BooleanField(default=False)
