from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail
from classify.utils.ai_model_training import get_pretrained_model, get_dataset, get_trainer

@shared_task
def send_verification_email(user_email, code):
  send_mail(
    subject='Email Verification',
    message=f'Your verification code is: {code}',
    from_email=settings.EMAIL_HOST_USER,
    recipient_list=[user_email],
    fail_silently=False
  )

def prepare_dataset(path_to_dataset, training_size):
  dataset_split = 0.1 if training_size is None else training_size
  return get_dataset(path_to_dataset, dataset_split)

def prepare_pretrained_model(ds):
  labels = ds['train'].features['label'].names
  return get_pretrained_model(labels)

@shared_task
def train_and_save_model(path_to_dataset, training_size):
  ds = prepare_dataset(path_to_dataset, training_size)
  model = prepare_pretrained_model(ds)
  trainer = get_trainer(model, ds)
  trainer_results = trainer.train()
  trainer.save_model('{}/{}'.format(settings.MEDIA_ROOT, "model"))
  return trainer_results

def train_model(path_to_dataset, training_size):
  return train_and_save_model.delay(path_to_dataset, training_size)
