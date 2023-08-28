from django.core.files.storage import default_storage
from io import BytesIO
from PIL import Image
from zipfile import ZipFile, is_zipfile

def validate_image_zip(zip_file, allowed_extensions=['.png', '.jpg', '.jpeg']) -> bool:
  if not is_zipfile(zip_file): return False

  image_count = 0
  with ZipFile(zip_file, 'r') as zip_ref:
    for file_name in zip_ref.namelist():
      if file_name.endswith('/'):
        continue
      image_count += 1
      if image_count > 100:
        return False
      if not any(file_name.lower().endswith(ext) for ext in allowed_extensions):
        return False
      
      try:
        with zip_ref.open(file_name) as file:
          im = Image.open(BytesIO(file.read()))
          im.verify()
      except:
        return False

  return True

def handle_saving_uploaded_zip(save_path, zip_file):
  default_storage.save(save_path, zip_file)
