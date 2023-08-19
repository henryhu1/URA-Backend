from django.conf import settings
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from zipfile import ZipFile
from classify.forms import UploadAndTrainForm
from classify.static_image_classifier import StaticImageClassifier
from classify.tasks import train_model

@csrf_exempt
def single(request):
  if request.method != "POST":
    # TODO string constant
    return JsonResponse({"error": "Please provide image file"})

  image_file = request.FILES.get('image')
  if image_file is None:
    return JsonResponse({"error": "Please provide image file"})

  result = StaticImageClassifier.classify_image(image_file)
  return JsonResponse(result, safe=False)

@csrf_exempt
def upload_and_train(request):
  if request.method != "POST":
    # TODO string constant
    return JsonResponse({"error": ""})

  form = UploadAndTrainForm(request.POST)
  if not form.is_valid():
    # TODO string constant
    return JsonResponse({"error": ""})

  training_size = form.cleaned_data["training_size"]
  dataset_zip = request.FILES.get("dataset")
  if dataset_zip is None:
    return

  zip_name = dataset_zip.name
  default_storage.save(zip_name, dataset_zip)

  path_to_zip = '{}/{}'.format(settings.MEDIA_ROOT, zip_name)
  path_to_dataset = '{}/{}'.format(settings.MEDIA_ROOT, zip_name.split(".")[0])
  with ZipFile(path_to_zip) as zObject:
    zObject.extractall(path_to_dataset)
 
  training_results = train_model(path_to_dataset, training_size)

  # trainer.log_metrics("train", train_results.metrics)
  # trainer.save_metrics("train", train_results.metrics)
  # trainer.save_state()
  # metrics = trainer.evaluate(prepared_ds['validation'])
  # trainer.log_metrics("eval", metrics)
  # trainer.save_metrics("eval", metrics)
  # trainer.save_model("./vision-transformer-saved")

  return JsonResponse(zip_name, safe=False)
