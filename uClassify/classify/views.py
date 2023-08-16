from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from datasets import load_dataset
from zipfile import ZipFile
from .image_classifier import ImageClassifier

@csrf_exempt
def single(request):
  if request.method != "POST":
    # TODO string constant
    return JsonResponse({"error": "Please provide image file"})

  image_file = request.FILES.get('image')
  if image_file is None:
    return JsonResponse({"error": "Please provide image file"})

  image = Image.open(image_file)
  image = image.resize((224, 224))
  classifier = ImageClassifier()
  result = classifier.classify_image(image)
  return JsonResponse(result, safe=False)

@csrf_exempt
def upload_and_train(request):
  if request.method != "POST":
    # TODO string constant
    return JsonResponse({"error": ""})

  dataset_zip = request.FILES.get('dataset')
  if dataset_zip is None:
    return

  path_to_file = dataset_zip.name
  default_storage.save(path_to_file, dataset_zip)

  with ZipFile('{}/{}'.format(settings.MEDIA_ROOT, path_to_file)) as zObject:
    zObject.extractall(settings.MEDIA_ROOT)
  

  ds = load_dataset('./train_and_validate/')
  labels = ds['train'].features['label'].names

  classifier = ImageClassifier()
  def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = classifier.tokenizer([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs

  prepared_ds = ds.with_transform(transform)

  trainer = classifier.getTrainer
  train_results = trainer.train()
  trainer.save_model()
  trainer.log_metrics("train", train_results.metrics)
  trainer.save_metrics("train", train_results.metrics)
  trainer.save_state()

  metrics = trainer.evaluate(prepared_ds['validation'])
  trainer.log_metrics("eval", metrics)
  trainer.save_metrics("eval", metrics)

  metrics = trainer.evaluate(prepared_ds['validation'])

  trainer.save_model("./vision-transformer-saved")

  return JsonResponse(path_to_file, safe=False)
