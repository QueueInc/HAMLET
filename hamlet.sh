#!/bin/bash

docker stop hamlet
docker rm hamlet
docker build -t hamlet .
docker run --name hamlet --volume /tmp/.X11-unix:/tmp/.X11-unix --volume ${1}:/home/resources -e DISPLAY=${DISPLAY} -t hamlet


# ./hamlet.sh /mnt/c/Users/giuseppe.pisano5/Documents/MyProjects/HAMLET/resources