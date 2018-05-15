#!/bin/bash
#
# Purpose: Evaluate a trained model while it is being trained. The eval job will periodically poll the train directory
# for new checkpoints and evaluate them on a test dataset.
source activate DL-TFOD-API
cd /Users/geoff/Documents/Development_libraries/models/research

python object_detection/eval.py --logtostderr \
--pipeline_config_path=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/SSD-vehicles.config \
--eval_dir=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/eval \
--checkpoint_dir=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/training