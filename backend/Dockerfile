FROM python:3.9

WORKDIR /src

ADD requirements.txt .

COPY ./config.ini /src/config.ini

RUN apt-get update && apt-get install -y netcat

RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN pip install torch torchvision

COPY ./app /src/app

copy ./model_repository /src/model_repository

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]