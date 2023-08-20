from django.urls import path
from . import views

urlpatterns = [
    path("", views.single, name="single"),
    path("upload_and_train/", views.upload_and_train, name="upload_and_train"),
    path("customized_classifier/", views.customized_classifier, name="customized_classifier"),
]
