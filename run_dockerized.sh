docker build -t ba-ml-fs17 .
docker exec vongrdir-weilemar-ba-ml-fs17 "pip install cython"
docker exec vongrdir-weilemar-ba-ml-fs17 "pip install -r /BA-ML-FS17/requirements.txt"
nvidia-docker run -it -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 "$@"
