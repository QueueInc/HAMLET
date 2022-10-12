FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9
RUN apt-get update && \
    apt-get install -y git --no-install-recommends && \
    apt install default-jre && \   
    apt-get install -y openjdk-8-jre-headless  
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    pip install requests && \
    pip install tabulate && \
    pip install future && \
    pip uninstall h2o && \
    pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o && \
    rm requirements.txt
COPY automl /home/automl
WORKDIR /home
