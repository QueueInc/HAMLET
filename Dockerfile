FROM python:3.12
RUN apt-get update && apt-get install swig -y
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt
COPY automl /home/automl
WORKDIR /home
