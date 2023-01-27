FROM python:3.9

WORKDIR /src

ADD requirements.txt .

RUN apt-get update && apt-get install -y netcat

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY ./content /src/content

COPY ./app /src/app

CMD ["streamlit", "run", "--theme.base", "dark", "app/main.py"]