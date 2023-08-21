from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from zipfile import ZipFile
from classify.constants import CommonStrings, ErrorMessages
from classify.forms import UploadAndTrainForm, RegistrationForm, VerifyEmailForm
from classify.models import CustomUser, CustomizedImageClassificationModel, EmailVerification
from classify.tasks import train_model
from classify.utils.image_classifying import classify_image

@csrf_exempt
def register_user(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  form = RegistrationForm(request.POST)
  if form.is_valid():
    email = form.cleaned_data['email']
    password = form.cleaned_data['password']
    try:
      user = CustomUser.objects.create_user(email=email, password=password)
    except Exception:
      return JsonResponse({"error": ErrorMessages.CREATE_ACCOUNT_FAIL}, status=500)
    request.session['user_email'] = user.email
    return JsonResponse({"Account Creation": CommonStrings.SUCCESS})
  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@csrf_exempt
def resend_email_verification(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

@csrf_exempt
def verify_email(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  form = VerifyEmailForm(request.POST)
  if form.is_valid():
    user_email = request.session.get('user_email')
    if not user_email:
      return JsonResponse({'error': 'No email found in session.'}, status=400)
    code = form.cleaned_data['code']
    try:
      user = CustomUser.objects.get(email=user_email)
      email_ver = EmailVerification.objects.get(user=user)
      if code == email_ver.verification_code:
        email_ver.is_verified = True
        email_ver.save()
        return JsonResponse({"Account verification": CommonStrings.SUCCESS})
      else:
        return JsonResponse({"error": ErrorMessages.INCORRECT_VERIFICATION_CODE}, status=500)
    except CustomUser.DoesNotExist:
      return JsonResponse({"error": ErrorMessages.VERIFY_ACCOUNT_FAIL}, status=400)
    except EmailVerification.DoesNotExist:
      return JsonResponse({"error": ErrorMessages.VERIFY_ACCOUNT_FAIL}, status=400)
  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@csrf_exempt
def authenticate_user(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  login_attempt_username = request.POST.get('username')
  if request.user.is_authenticated:
    if request.user.username != login_attempt_username:
      request.session.flush()

  form = AuthenticationForm(request, data=request.POST)
  if form.is_valid():
    user = form.get_user()
    login(request, user)
    user = CustomUser.objects.get(email=user.email)
    email_ver = EmailVerification.objects.get(user=user)
    return JsonResponse({"is_verified": email_ver.is_verified})
  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@csrf_exempt
def my_account(request):
  if request.method != "GET":
    return JsonResponse({"error": ErrorMessages.ONLY_GET}, status=405)
  
  user = request.user
  if user.is_authenticated:
    email_ver = EmailVerification.objects.get(user=user)
    return JsonResponse({"is_verified": email_ver.is_verified})
  else:
    return JsonResponse({"error": ErrorMessages.UNAUTHORIZED_ACCESS}, status=401)

@csrf_exempt
def logout_user(request):
  if request.method != 'POST':
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  logout(request)
  return JsonResponse({"status": "success", "message": "Logged out successfully"})

@csrf_exempt
def single(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  image_file = request.FILES.get('image')
  if image_file is None:
    return JsonResponse({"error": ErrorMessages.PROVIDE_IMAGE})

  result = classify_image(image_file)
  return JsonResponse(result, safe=False)

@csrf_exempt
def upload_and_train(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  form = UploadAndTrainForm(request.POST)
  if not form.is_valid():
    # TODO string constant
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

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

@csrf_exempt
@login_required
def customized_classifier(request):
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  image_file = request.FILES.get('image')
  if image_file is None:
    return JsonResponse({"error": ErrorMessages.PROVIDE_IMAGE})

  #TODO make it session user
  user = CustomUser.objects.get(pk=1)
  customized_classifier = CustomizedImageClassificationModel.objects.get(owner=user)
  print(user)
  print(customized_classifier)

  result = classify_image(image_file, customized_classifier.model_path)
  return JsonResponse(result, safe=False)
