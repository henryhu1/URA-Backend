from celery.result import AsyncResult
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import AnonymousUser
from django.contrib.auth import login, logout
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from zipfile import ZipFile
from classify.constants import CommonStrings, ErrorMessages
from classify.forms import UploadAndTrainForm, RegistrationForm, VerifyEmailForm
from classify.models import CustomUser, CustomizedImageClassificationModel, EmailVerification, TrainingModelTask
from classify.tasks import train_model
from classify.utils.image_classifying import classify_image
from classify.utils.jwt_utils import create_access_token, create_refresh_token

@ensure_csrf_cookie
def get_csrf_token(request: HttpRequest) -> JsonResponse:
  return JsonResponse({"detail": "CSRF cookie set"})

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def register_user(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  form = RegistrationForm(request.POST)
  if form.is_valid():
    email = form.cleaned_data['email']
    username = form.cleaned_data['username']
    password = form.cleaned_data['password']

    try:
      user = CustomUser.objects.create_user(email=email, username=username, password=password)
      login(request, user)
    except Exception:
      return JsonResponse({"error": ErrorMessages.CREATE_ACCOUNT_FAIL}, status=500)

    return JsonResponse({"Account Creation": CommonStrings.SUCCESS})

  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@login_required
@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def resend_email_verification(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  requesting_user = request.user
  user_email = requesting_user.user_email
  if not user_email:
    return JsonResponse({'error': 'No email found in session.'}, status=400)

  #TODO send another verification email

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def verify_email(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  form = VerifyEmailForm(request.POST)
  if form.is_valid():
    requesting_user = request.user
    if isinstance(requesting_user, AnonymousUser):
      return JsonResponse({'error': 'No user found in session.'}, status=400)

    user_email = requesting_user.user_email
    if not user_email:
      return JsonResponse({'error': 'No email found in session.'}, status=400)

    code = form.cleaned_data['code']
    try:
      email_ver = EmailVerification.objects.get(user=requesting_user)
      if code == email_ver.verification_code:
        email_ver.is_verified = True
        email_ver.save()
        refresh_token = create_refresh_token(requesting_user)
        access_token = create_access_token(refresh_token)
        return JsonResponse({"access_token": str(access_token), "refresh_token": str(refresh_token)})
      else:
        return JsonResponse({"error": ErrorMessages.INCORRECT_VERIFICATION_CODE}, status=500)
    except CustomUser.DoesNotExist:
      return JsonResponse({"error": ErrorMessages.VERIFY_ACCOUNT_FAIL}, status=400)
    except EmailVerification.DoesNotExist:
      return JsonResponse({"error": ErrorMessages.VERIFY_ACCOUNT_FAIL}, status=400)

  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def authenticate_user(request: HttpRequest) -> JsonResponse:
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

    try:
      email_ver = EmailVerification.objects.get(user=user)
      if email_ver.is_verified:
        refresh_token = create_refresh_token(user)
        access_token = create_access_token(refresh_token)
        return JsonResponse({"is_verified": email_ver.is_verified, "access_token": str(access_token), "refresh_token": str(refresh_token)})
      else:
        return JsonResponse({"is_verified": email_ver.is_verified})
    except CustomUser.DoesNotExist:
      return JsonResponse({"error": "User not found"}, status=404)
    except EmailVerification.DoesNotExist:
      return JsonResponse({"error": "Email verification record not found"}, status=404)

  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@login_required
@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def my_account(request: HttpRequest) -> JsonResponse:
  if request.method != "GET":
    return JsonResponse({"error": ErrorMessages.ONLY_GET}, status=405)

  requesting_user = request.user
  if requesting_user.is_authenticated:
    try:
      email_ver = EmailVerification.objects.get(user=requesting_user)
      is_verified = email_ver.is_verified
      # if not is_verified:
      return JsonResponse({"is_verified": is_verified})
      
      # custom_classifier = CustomizedImageClassificationModel.objects.get(owner=requesting_user)
      # return JsonResponse({"is_verified": is_verified})
    except EmailVerification.DoesNotExist:
      return JsonResponse({"error": "Email verification record not found"}, status=404)
    # except CustomizedImageClassificationModel.DoesNotExist:
    #   return JsonResponse({"error": "Customized model not found"}, status=404)

  else:
    return JsonResponse({"error": ErrorMessages.UNAUTHORIZED_ACCESS}, status=401)

@login_required
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def logout_user(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  logout(request)
  return JsonResponse({"status": "success", "message": "Logged out successfully"})

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def single(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  image_file = request.FILES.get('image')
  if image_file is None:
    return JsonResponse({"error": ErrorMessages.PROVIDE_IMAGE})

  result = classify_image(image_file)
  return JsonResponse(result, safe=False)

@login_required
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def upload_and_train(request: HttpRequest) -> JsonResponse:
  #TODO split method
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  requesting_user = request.user
  if not requesting_user.is_authenticated:
    return JsonResponse({"error": ErrorMessages.UNAUTHORIZED_ACCESS}, status=401)

  has_customized_model = CustomizedImageClassificationModel.objects.filter(owner=requesting_user)
  if has_customized_model.exists():
    return JsonResponse({"error": "You already have a customized model."}, status=400)

  has_task = TrainingModelTask.objects.filter(owner=requesting_user)
  if has_task.exists() and not AsyncResult(has_task.first().task_id).ready():
    return JsonResponse({"error": "You already have a customized model"}, status=400)

  form = UploadAndTrainForm(request.POST)
  if not form.is_valid():
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

  training_size = form.cleaned_data["training_size"]
  dataset_zip = request.FILES.get("dataset")
  if dataset_zip is None:
    return

  # print(dataset_zip.name)
  # print("".join(dataset_zip.name.split(".").pop()))
  # default_storage.save(zip_name, dataset_zip)
  new_model = CustomizedImageClassificationModel.objects.create(owner=requesting_user)
  new_model.save()

  # path_to_zip = '{}/{}'.format(settings.MEDIA_ROOT, zip_name)
  # path_to_dataset = '{}/{}'.format(settings.MEDIA_ROOT, zip_name.split(".")[0])
  with ZipFile(dataset_zip) as zObject:
    # zObject.extractall(path_to_dataset)
    zObject.extractall(new_model.path_to_dataset())
 
  train_model(requesting_user, new_model, training_size)
  # task_id = training_task_celery.task_id

  # training_task = TrainingModelTask.objects.create(owner=requesting_user, model=new_model, task_id=task_id)
  # training_task.save()

  return JsonResponse(new_model.path_to_dataset(), safe=False)

@login_required
@api_view(['POST'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def customized_classifier(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  image_file = request.FILES.get('image')
  if image_file is None:
    return JsonResponse({"error": ErrorMessages.PROVIDE_IMAGE})

  requesting_user = request.user
  customized_classifier = CustomizedImageClassificationModel.objects.get(owner=requesting_user)

  result = classify_image(image_file, customized_classifier.model_path)
  return JsonResponse(result, safe=False)
