# URA-Backend
Backend for image classification tool using **Django**, **Celery** and **RabbitMQ**.

Email huhenry50@gmail.com for any questions.

To interact with the server, make sure the [frontend](https://github.com/henryhu1/URA-Frontend) is running as well.

<details open>
<summary>Table of Contents</summary>

  - [Local Usage](#local-usage)
    - [With Docker](#with-docker)
    - [Without Docker](#without-docker)
  - [Production Environment](#production-environment)
  - [TODO](#todo)
</details>

## Local Usage
The server can be run using [Docker](#with-docker) and [without Docker](#without-docker).

### With Docker
The images are built **locally**.
For the curious, details are in [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml).

- #### 1. First Time Setup
  From a new terminal, in your local cloned repository directory, execute:
  ```shell
  $ docker-compose build
  ```

- #### 2. Run Server
  ```shell
  $ docker-compose up -d
  $ docker-compose run web manage.py makemigrations
  $ docker-compose run web manage.py migrate
  ```

- #### 3. Stop Server
  ```shell
  $ docker-compose down
  ```

### Without Docker
If you do not wish to use the Docker image, this section provides instructions on how to run the server manually.

*Note: this section still uses Docker for RabbitMQ. You can follow [the below RabbitMQ setup](#rabbitmq) or set up RabbitMQ locally without a RabbitMQ image.*

#### Environment Variables
In your local repo, create a `.env` file and place these contents inside:
```properties
PRODUCTION=False
DEBUG=True

DB_ENGINE=django.db.backends.sqlite3
DB_NAME={place your desired sqlite file path here}

# Frontend host and port
CORS_ALLOWED_ORIGINS=http://localhost:3000
 
# OPTIONAL email configuration, check the SMTP servers and ports 
EMAIL_HOST={email server host here, ex. smtp.gmail.com} 
EMAIL_PORT={email server port here, ex. 587}
EMAIL_HOST_USER={your email here}
EMAIL_HOST_PASSWORD={your email password here}

# RabbitMQ host and port
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672//
```

***BE CAREFUL: DO NOT SHARE THIS*** `.env` ***FILE.***

#### First Time Setup
This setup uses `pipenv` as the Python virtual environment.

From a new terminal, in your local cloned repository directory, execute:
```shell
$ pipenv shell
$ pipenv install
$ cd uClassify/
$ python manage.py makemigrations
$ python manage.py migrate
```

#### Run Server
Continuing from above, execute:
```shell
$ python manage.py runserver
```
For every fresh run, execute:
```shell
$ pipenv shell
$ cd uClassify/
$ python manage.py runserver
```

#### RabbitMQ
By default, the RabbitMQ image is used by celery.
You can pull any [RabbitMQ image](https://hub.docker.com/_/rabbitmq) you like from Docker Hub.

ex.
```shell
$ docker pull rabbitmq
```

Then run RabbitMQ (either through the Docker desktop app or from your command line).

***IMPORTANT: make sure the [CELERY_BROKER_URL](#environment-variables) is pointing to the RabbitMQ port, by default it is 5672***

#### Celery Worker
From a new terminal, in your local cloned repository directory, execute:
```shell
$ pipenv shell
$ cd uClassify
$ celery -A uClassify worker -l INFO -P solo
```

## Production Environment
***The EC2 instance is currently down.***

AWS powers the server hosted on my domain, utilizing EC2, RDS and S3.

## TODO
- code cleanup
- improve error handling
- (prod) provide classification model downloads
 