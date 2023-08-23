from django.conf import settings
from django.contrib.auth.models import AbstractUser, BaseUserManager, PermissionsMixin
from django.core.exceptions import ValidationError
from django.core.mail import send_mail
from django.core.validators import validate_email
from django.db import models
import dns.resolver
import random

class CustomUserManager(BaseUserManager):
  def validate_email_domain(self, email):
    domain = email.split('@')[1]
    try:
      dns.resolver.resolve(domain, 'MX')
      return True
    except dns.resolver.NoAnswer:
      return False
    except dns.resolver.NXDOMAIN:
      return False
    except Exception as e:
      return False

  def create_user(self, email, password=None, **extra_fields):
    if not email:
      raise ValueError('The Email field must be set')

    #TODO encapsulate verification email
    try:
      validate_email(email)
    except ValidationError:
      raise ValueError('Invalid email format')

    if not self.validate_email_domain(email):
      raise ValueError('The email domain does not have valid MX records')

    email = self.normalize_email(email)
    user = self.model(email=email, **extra_fields)
    user.set_password(password)
    user.save(using=self._db)

    if not settings.DEBUG:
      code = ''.join(random.choices('0123456789', k=6))
      EmailVerification.objects.create(user=user, verification_code=code)

      send_mail(
        subject='Email Verification',
        message=f'Your verification code is: {code}',
        from_email=settings.EMAIL_HOST_USER,
        recipient_list=[user.email],
        fail_silently=False
      )
    else:
      EmailVerification.objects.create(user=user, is_verified=True)

    return user

  def create_superuser(self, email, password=None, **extra_fields):
    extra_fields.setdefault('is_staff', True)
    extra_fields.setdefault('is_superuser', True)

    return self.create_user(email, password, **extra_fields)

class CustomUser(AbstractUser):
  email = models.EmailField(unique=True)
  date_joined = models.DateTimeField(auto_now_add=True)
  is_active = models.BooleanField(default=True)
  is_staff = models.BooleanField(default=False)
  
  objects = CustomUserManager()

  USERNAME_FIELD = 'email'
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
  is_done = models.BooleanField(default=False)

class EmailVerification(models.Model):
  user = models.OneToOneField(CustomUser, on_delete=models.CASCADE)
  verification_code = models.CharField(max_length=6)
  is_verified = models.BooleanField(default=False)
