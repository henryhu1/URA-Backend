from django import forms
from classify.models import CustomUser

class UploadAndTrainForm(forms.Form):
  training_size = forms.FloatField(
    label="training_size",
    max_value=1.0,
    min_value=0.0,
    required=False
  )

class RegistrationForm(forms.ModelForm):
  class Meta:
    model = CustomUser
    fields = ['email', 'username', 'password']

class VerifyEmailForm(forms.Form):
  code = forms.CharField(max_length=6)
