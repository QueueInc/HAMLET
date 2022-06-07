FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9
RUN apt-get update && apt-get install -y git --no-install-recommends
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt
COPY automl /home/automl
WORKDIR /home
COPY run_experiments.sh run_experiments.sh
