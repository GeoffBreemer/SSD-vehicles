#!/bin/bash
#
# Purpose: Export a trained model to a TensorFlow model.
source activate DL-TFOD-API
cd /Users/geoff/Documents/Development_libraries/models/research

python object_detection/export_inference_graph.py --input_type image_tensor \
--pipeline_config_path=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/SSD-vehicles.config \
--trained_checkpoint_prefix=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/training/model.ckpt-89 \
--output_directory=/Users/geoff/Documents/Development/object-detection/SSD-vehicles/models/model/export

cd /Users/geoff/Documents/Development/object-detection/SSD-vehicles
