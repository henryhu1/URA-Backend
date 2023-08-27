from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail
from classify.models import CustomizedImageClassificationModel, EmailVerification, TrainingModelTask
from classify.utils.ai_model_training import get_pretrained_model, get_dataset, get_trainer, get_training_and_validation_datasets, get_hub_model
import random
import os

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
    send_verification_email(user.email)
  else:
    EmailVerification.objects.create(user=user, is_verified=True)

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
def train_and_save_model(user_id, model_id, model_type, training_size):
  model = CustomizedImageClassificationModel.objects.get(pk=model_id)
  path_to_dataset = model.path_to_dataset()
  dataset_split = 0.1 if training_size is None else training_size
  if model_type == CustomizedImageClassificationModel.VISION_TRANSFORMER:
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
    dataset_directory = [d for d in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset, d))][0]
    dataset_directory = os.path.join(path_to_dataset, dataset_directory)
    ds = get_training_and_validation_datasets(dataset_directory, dataset_split)
    train_ds = ds[0]
    val_ds = ds[1]
    class_names = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]
    pretrained_model = get_hub_model(CustomizedImageClassificationModel.TENSORFLOW_HUB_URLS[model_type], len(class_names))
    history = pretrained_model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=10
    )
    model_path = '{}/{}/{}'.format(settings.MEDIA_ROOT, user_id, "model.keras")
    pretrained_model.save(model_path)
    model.model_path = model_path
    model.save()

def handle_model_training(user, model, training_size):
  res = train_and_save_model.apply_async((user.id, model.id, model.model_type, training_size), link=send_finished_model_training_email.si(user.email))

  training_task = TrainingModelTask.objects.create(owner=user, model=model, task_id=res.task_id)
  training_task.save()
  return res.task_id
