import os

# Path to the original data set
PATH_BASE = "../../datasets/dlib_front_and_rear_vehicles_v1"
PATH_XML_TRAIN = os.path.sep.join([PATH_BASE, "training.xml"])
PATH_XML_TEST = os.path.sep.join([PATH_BASE, "testing.xml"])

# Paths to the TFOD records and label map file
PATH_RECORDS = "./data"
PATH_TRAIN_RECORD = os.path.sep.join([PATH_RECORDS, "training.record"])
PATH_TEST_RECORD = os.path.sep.join([PATH_RECORDS, "testing.record"])
PATH_LABEL_MAP = os.path.sep.join([PATH_RECORDS, "classes.pbtxt"])

CLASSES = {"rear": 1, "front": 2}       # always start at 1
