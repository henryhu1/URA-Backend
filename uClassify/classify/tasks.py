from celery import shared_task
from django.conf import settings
from django.core.files import File
from django.core.files.storage import default_storage
from django.core.mail import send_mail
from classify.models import CustomizedImageClassificationModel, EmailVerification, TrainingModelTask
from classify.utils.ai_model_training import get_pretrained_model, get_dataset, get_trainer, get_training_and_validation_datasets, get_hub_model
from zipfile import ZipFile
import random
import os
import shutil
import tempfile

@shared_task
def send_verification_email(user_email, code):
  send_mail(
    subject='Email Verification',
    message=f'Your verification code is: {code}',
    from_email=settings.EMAIL_HOST_USER,
    recipient_list=[user_email],
    fail_silently=False
  )

def handle_email_verification(user):
  if not settings.DEBUG:
    code = ''.join(random.choices('0123456789', k=6))
    EmailVerification.objects.create(user=user, verification_code=code)
    send_verification_email.delay(user.email, code)
  else:
    EmailVerification.objects.create(user=user, is_verified=True)

def handle_resend_email_verification(user_email, email_ver: EmailVerification):
  if not settings.DEBUG:
    code = ''.join(random.choices('0123456789', k=6))
    email_ver.verification_code = code
    email_ver.save()
    send_verification_email.delay(user_email, code)
  else:
    email_ver.is_verified=True
    email_ver.save()

@shared_task
def send_finished_model_training_email(user_email):
  send_mail(
    subject='Model Training',
    message='Your model is done being trained! Try classifying some images!',
    from_email=settings.EMAIL_HOST_USER,
    recipient_list=[user_email],
    fail_silently=False
  )

@shared_task
def train_and_save_model(user_id, model_id, training_size):
  model = CustomizedImageClassificationModel.objects.get(pk=model_id)
  model_type = model.model_type
  path_to_dataset = model.path_to_dataset()
  dataset_split = 0.1 if training_size is None else training_size
  if model_type == CustomizedImageClassificationModel.VISION_TRANSFORMER:
    return ''
    ds = get_dataset(path_to_dataset, dataset_split)
    labels = ds['train'].features['label'].names
    pretrained_model = get_pretrained_model(labels)
    trainer = get_trainer(pretrained_model, ds)
    trainer_results = trainer.train()
    model_path = '{}/{}/{}'.format(settings.MEDIA_ROOT, user_id, "model")
    trainer.save_model(model_path)
    model.model_path = model_path
    model.save()
    return trainer_results
  else:
    temp_dataset_directory = tempfile.mkdtemp()
    with ZipFile(path_to_dataset, 'r') as zip_ref:
      zip_ref.extractall(temp_dataset_directory)
    dataset_directory = [d for d in os.listdir(temp_dataset_directory) if os.path.isdir(os.path.join(temp_dataset_directory, d))][0]
    dataset_directory = os.path.join(temp_dataset_directory, dataset_directory)
    ds = get_training_and_validation_datasets(dataset_directory, dataset_split)
    train_ds = ds[0]
    val_ds = ds[1]
    class_names = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]
    model.labels_list = class_names
    pretrained_model = get_hub_model(CustomizedImageClassificationModel.TENSORFLOW_HUB_URLS[model_type], len(class_names))
    history = pretrained_model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=10
    )
    shutil.rmtree(temp_dataset_directory)
    temp_model_dir = tempfile.mkdtemp()
    temp_model_path = '{}/{}'.format(temp_model_dir, 'model.keras')
    pretrained_model.save(temp_model_path)
    with open(temp_model_path, 'rb') as f:
      default_storage.save(model.path_to_model(), File(f))
    model.model_path = model.path_to_model()
    model.save()
    shutil.rmtree(temp_model_dir)

def handle_model_training(user, model, training_size):
  res = train_and_save_model.apply_async((user.id, model.id, model.model_type, training_size), link=send_finished_model_training_email.si(user.email))

  training_task = TrainingModelTask.objects.create(owner=user, model=model, task_id=res.task_id)
  training_task.save()
  return res.task_id
