FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9
RUN apt-get update && apt-get install -y default-jdk gedit --no-install-recommends
COPY .devcontainer/requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install black && \
    pip install --no-cache-dir --upgrade -r /requirements.txt && \
    rm requirements.txt
COPY automl /automl
COPY argumentation /argumentation
RUN cd /argumentation && ./gradlew shadowJar
WORKDIR /home
ENTRYPOINT ["java", "-jar", "/argumentation/build/libs/hamlet-1.0-SNAPSHOT-all.jar", "/home/resources", "true"]
