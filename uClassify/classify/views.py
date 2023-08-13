from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from .image_classifier import ImageClassifier

@csrf_exempt
def single(request):
  if request.method == "POST":
    if request.FILES.get('image'):
      image_file = request.FILES.get('image')
      image = Image.open(image_file)
      image = image.resize((224, 224))
      classifier = ImageClassifier()
      result = classifier.classify_image(image)
      return JsonResponse(result, safe=False)
  # TODO string constant
  return JsonResponse({"error": "Please provide image file"})
