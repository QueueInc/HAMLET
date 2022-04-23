FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9
RUN apt-get update && apt-get install -y git default-jre default-jdk gedit --no-install-recommends
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt

