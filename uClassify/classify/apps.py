from django.apps import AppConfig
from classify.static_image_classifier import StaticImageClassifier


class ClassifyConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classify'

    def ready(self):
        StaticImageClassifier.get_model()
        StaticImageClassifier.get_tokenizer()
