from django.urls import path
from . import views

urlpatterns = [
    path("", views.single, name="single"),
    path("upload_and_train/", views.upload_and_train, name="upload_and_train"),
    path("customized_classifier/", views.customized_classifier, name="customized_classifier"),
    path("delete_custom_model/", views.delete_custom_model, name="delete_custom_model"),
    path("register_user/", views.register_user, name="register_user"),
    path("resend_email_verification/", views.resend_email_verification, name="resend_email_verification"),
    path("authenticate_user/", views.authenticate_user, name="authenticate_user"),
    path("my_account/", views.my_account, name="my_account"),
    path("logout/", views.logout_user, name="logout"),
    path("get_csrf_token/", views.get_csrf_token, name="get_csrf_token"),
]
