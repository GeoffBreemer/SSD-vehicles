#!/bin/bash
#
# Purpose: Train a model. The training job runs indefinitely until it is killed.
source activate DL-TFOD-API
cd /Users/geoff/Documents/Development_libraries/models/research

python object_detection/train.py --logtostderr \
--pipeline_config_path=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/SSD-vehicles.config \
--train_dir=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/training
