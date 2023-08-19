from django import forms

class UploadAndTrainForm(forms.Form):
  training_size = forms.FloatField(
    label="training_size",
    max_value=1.0,
    min_value=0.0,
    required=False
  )
