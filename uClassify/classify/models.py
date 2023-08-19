from django.contrib.auth.models import AbstractUser, BaseUserManager, PermissionsMixin
from django.core.validators import validate_email
from django.core.exceptions import ValidationError
from django.db import models
import dns.resolver

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

    # Basic email format validation
    try:
      validate_email(email)
    except ValidationError:
      raise ValueError('Invalid email format')

    # Check if the email domain has MX records
    if not self.validate_email_domain(email):
      raise ValueError('The email domain does not have valid MX records')

    email = self.normalize_email(email)
    user = self.model(email=email, **extra_fields)
    user.set_password(password)
    user.save(using=self._db)
    return user

  def create_superuser(self, email, password=None, **extra_fields):
    extra_fields.setdefault('is_staff', True)
    extra_fields.setdefault('is_superuser', True)

    return self.create_user(email, password, **extra_fields)

class CustomUser(AbstractUser, PermissionsMixin):
  email = models.EmailField(unique=True)
  first_name = models.CharField(max_length=30, blank=True)
  last_name = models.CharField(max_length=30, blank=True)
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
  dataset_path = models.FileField(upload_to='uploads/')
  model_path = models.FileField(upload_to='models/')
