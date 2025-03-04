#!/bin/bash


CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker run -it \
  --network host \
  --gpus all \
  --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v $CURRENT_DIR:/home/duckietown_rl_course/$(basename "$CURRENT_DIR") \
  --name duckietownrl_container \
  duckietownrl:latest  \
  bash
