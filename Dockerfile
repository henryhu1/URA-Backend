# FROM python:3.9

# ENV PYTHONUNBUFFERED 1
# ENV DJANGO_SETTINGS_MODULE uClassify.settings

# WORKDIR /uClassify

# COPY requirements.txt /uClassify/
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . /uClassify/

# RUN python manage.py collectstatic --noinput

# CMD ["gunicorn", "yourapp.wsgi:application", "--bind", "0.0.0.0:8000"]

FROM python:3.10.13

WORKDIR /uClassify

COPY requirements.txt /uClassify/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /uClassify/

EXPOSE 8000

WORKDIR /uClassify/uClassify

ENTRYPOINT ["python"]
CMD ["manage.py", "runserver", "0.0.0.0:8000"]
