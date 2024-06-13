FROM python:3.9
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt
COPY resources/datasets /home/resources/datasets
COPY automl /home/automl
WORKDIR /home
