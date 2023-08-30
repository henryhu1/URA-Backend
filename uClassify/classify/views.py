from celery.result import AsyncResult
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from classify.constants import CommonStrings, ErrorMessages
from classify.forms import UploadAndTrainForm, RegistrationForm, AuthenticateUserForm
from classify.models import CustomUser, CustomizedImageClassificationModel, EmailVerification, TrainingModelTask
from classify.tasks import handle_model_training, handle_email_verification, handle_resend_email_verification
from classify.utils.image_classifying import classify_image
from classify.utils.image_uploading import validate_image_zip, handle_saving_uploaded_zip
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
      handle_email_verification(user)
      login(request, user)
    except Exception:
      return JsonResponse({"error": ErrorMessages.CREATE_ACCOUNT_FAIL}, status=500)

    return JsonResponse({"Account Creation": CommonStrings.SUCCESS})

  else:
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def resend_email_verification(request: HttpRequest) -> JsonResponse:
  if request.method != "POST":
    return JsonResponse({"error": ErrorMessages.ONLY_POST}, status=405)

  request_username = request.POST.get('username')
  try:
    existing_user = CustomUser.objects.filter(username=request_username)
    if not existing_user.exists():
      return JsonResponse({"error": f"User with the username {request_username} not found"}, status=404)
    
    user = existing_user.first()
    user_email = user.email
    if not user_email:
      return JsonResponse({'error': 'No email found.'}, status=404)

    email_ver = EmailVerification.objects.get(user=user)
    handle_resend_email_verification(user_email, email_ver)
  except EmailVerification.DoesNotExist:
    return JsonResponse({"error": ErrorMessages.VERIFY_ACCOUNT_FAIL}, status=400)

  return JsonResponse({"status": CommonStrings.SUCCESS})

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

  form = AuthenticateUserForm(request, data=request.POST)
  if form.is_valid():
    user = form.get_user()
    login(request, user)

    try:
      email_ver = EmailVerification.objects.get(user=user)
      if not email_ver.is_verified:
        code = form.cleaned_data['code']
        if code == email_ver.verification_code:
          email_ver.is_verified = True
          email_ver.save()
        else:
          return JsonResponse({"is_verified": email_ver.is_verified})
    except CustomUser.DoesNotExist:
      return JsonResponse({"error": "User not found"}, status=404)
    except EmailVerification.DoesNotExist:
      return JsonResponse({"error": "Email verification record not found"}, status=404)

    refresh_token = create_refresh_token(user)
    access_token = create_access_token(refresh_token)
    return JsonResponse({"is_verified": email_ver.is_verified, "access_token": str(access_token), "refresh_token": str(refresh_token)})

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
    returning_json = {}
    try:
      email_ver = EmailVerification.objects.get(user=requesting_user)
      is_verified = email_ver.is_verified
      returning_json["is_verified"] = is_verified
      
      has_task = TrainingModelTask.objects.filter(owner=requesting_user)
      if has_task.exists():
        returning_json["running_task"] = not AsyncResult(has_task.first().task_id).ready()

      has_customized_model = CustomizedImageClassificationModel.objects.filter(owner=requesting_user)
      returning_json["existing_model"] = has_customized_model.exists()
      return JsonResponse(returning_json)
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
  return JsonResponse({"prediction": result})

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

  has_task = TrainingModelTask.objects.filter(owner=requesting_user)
  if has_task.exists() and not AsyncResult(has_task.first().task_id).ready():
    return JsonResponse({"error": "You have a model in training."}, status=400)

  has_customized_model = CustomizedImageClassificationModel.objects.filter(owner=requesting_user)
  if has_customized_model.exists():
    return JsonResponse({"error": "You already have a customized model."}, status=400)

  form = UploadAndTrainForm(request.POST)
  if not form.is_valid():
    return JsonResponse({"error": ErrorMessages.INVALID_FORM}, status=400)

  training_size = form.cleaned_data['training_size']
  model_type = form.cleaned_data['model_type']
  dataset_zip = request.FILES.get("dataset")
  if dataset_zip is None:
    return JsonResponse({"error": "Please upload a zip file."}, status=400)

  if dataset_zip.size > 5 * 1024 * 1024:
    return JsonResponse({"error": "Uploaded folder is too large. Please upload less training images."}, status=400)

  if model_type == CustomizedImageClassificationModel.VISION_TRANSFORMER:
    return JsonResponse({"error": "Vision Transformers are currently unavailable."}, status=400)

  if not validate_image_zip(dataset_zip):
    return JsonResponse({"error": "Please upload a valid folder containing only images."}, status=400)
  # path_to_zip = '{}/{}'.format(settings.MEDIA_ROOT, zip_name)
  # path_to_dataset = '{}/{}'.format(settings.MEDIA_ROOT, zip_name.split(".")[0])

  # print(dataset_zip.name)
  # print("".join(dataset_zip.name.split(".").pop()))
  # default_storage.save(zip_name, dataset_zip)
  new_model = CustomizedImageClassificationModel.objects.create(owner=requesting_user, model_type=model_type)

  handle_saving_uploaded_zip(new_model.path_to_dataset(), dataset_zip)
  new_model.dataset_path = new_model.path_to_dataset()
  new_model.save()
 
  handle_model_training(requesting_user, new_model, training_size)
  # task_id = training_task_celery.task_id

  # training_task = TrainingModelTask.objects.create(owner=requesting_user, model=new_model, task_id=task_id)
  # training_task.save()

  return JsonResponse({"status": CommonStrings.SUCCESS})

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

  has_customized_model = CustomizedImageClassificationModel.objects.filter(owner=requesting_user)
  if not has_customized_model.exists():
    return JsonResponse({"error": "You do not have a customized model."}, status=404)

  has_task = TrainingModelTask.objects.filter(owner=requesting_user)
  if has_task.exists() and not AsyncResult(has_task.first().task_id).ready():
    return JsonResponse({"error": "You currently have a model in training."}, status=400)

  customized_classifier = has_customized_model.first()
  result = classify_image(
    image_file,
    customized_classifier.model_type == CustomizedImageClassificationModel.VISION_TRANSFORMER,
    customized_classifier.model_path,
    customized_classifier.labels_list
  )
  return JsonResponse({"prediction": result})

@login_required
@api_view(['DELETE'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def delete_custom_model(request: HttpRequest) -> JsonResponse:
  if request.method != "DELETE":
    return JsonResponse({"error": ErrorMessages.ONLY_DELETE}, status=405)

  requesting_user = request.user
  if not requesting_user.is_authenticated:
    return JsonResponse({"error": ErrorMessages.UNAUTHORIZED_ACCESS}, status=401)

  has_customized_model = CustomizedImageClassificationModel.objects.filter(owner=requesting_user)
  has_task = TrainingModelTask.objects.filter(owner=requesting_user)
  if has_task.exists() and not AsyncResult(has_task.first().task_id).ready():
    return JsonResponse({"error": "Your model is training. You can delete the model once it has finished training."}, status=400)
  elif not has_customized_model.exists():
    return JsonResponse({"error": ErrorMessages.NOT_FOUND}, status=404)
  else:
    has_customized_model.first().delete()
    return JsonResponse({"status": CommonStrings.SUCCESS})
