xhost +local:root
DATA_PATH=~/
docker run \
    --gpus all \
    --privileged \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/udev:/etc/udev \
    -v /etc/udev:/etc/udev \
    -v /dev:/dev \
    -v $DATA_PATH:/ws \
    -v /dev:/dev \
    --network=host --name gello -it gello


