# URA-Backend
Backend for image classification tool using Django, Celery and RabbitMQ

## Celery
start the worker with
```
celery -A uClassify worker -l INFO -P solo
```
