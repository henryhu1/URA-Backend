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
  owner = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
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
