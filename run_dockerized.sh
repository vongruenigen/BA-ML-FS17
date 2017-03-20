docker build -t ba-ml-fs17 .
nvidia-docker run -it -v /cluster/home/weilemar/data/farm-ai/images/BA-ML-FS17:/BA-ML-FS17 \
              "pip install cython && pip install -r /BA-ML-FS17/requirements.txt && /BA-ML-FS17/run_seq2seq.sh"
