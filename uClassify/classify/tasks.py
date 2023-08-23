from celery import shared_task
from django.conf import settings
from django.core.mail import send_mail
from classify.models import CustomizedImageClassificationModel, TrainingModelTask
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
def train_and_save_model(user_id, model_id, training_size):
  model = CustomizedImageClassificationModel.objects.get(pk=model_id)
  print(model.id)
  ds = prepare_dataset(model.path_to_dataset(), training_size)
  pretrained_model = prepare_pretrained_model(ds)
  trainer = get_trainer(pretrained_model, ds)
  trainer_results = trainer.train()
  model_path = '{}/{}/{}'.format(settings.MEDIA_ROOT, user_id, "model")
  trainer.save_model(model_path)
  customized_model = CustomizedImageClassificationModel.objects.get(pk=model.id)
  customized_model.model_path = model_path
  customized_model.save()
  return trainer_results

def train_model(user, model, training_size):
  task = train_and_save_model.delay(user.id, model.id, training_size)

  training_task = TrainingModelTask.objects.create(owner=user, model=model, task_id=task.task_id)
  training_task.save()
  return task.task_id
