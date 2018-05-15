"""Convert the dlib vehicle database to TensorFlow records

More information: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
TFRecord information: https://www.tensorflow.org/api_guides/python/python_io#TFRecords_Format_Details
"""
import settings_ssd_vehicles as settings
from dltoolkit.utils.tfod import TFDataPoint

from bs4 import BeautifulSoup
from PIL import Image

import tensorflow as tf
import os
import progressbar


def main(_):
    # Create the classes.pbtxt file
    f = open(settings.PATH_LABEL_MAP, "w")

    for (k, v) in settings.CLASSES.items():
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: '" + k + "'\n"
                "}\n")
        f.write(item)

    f.close()

    # Convert the training and test data sets to data points
    datasets = [
        ("training set", settings.PATH_XML_TRAIN, settings.PATH_TRAIN_RECORD),
        ("test set", settings.PATH_XML_TEST, settings.PATH_TEST_RECORD)]

    # Loop over both data sets, read the images, their bounding box(es) and save them as data points
    for (dType, inputPath, outputPath) in datasets:
        contents = open(inputPath).read()

        # Prepare the XML reader
        soup = BeautifulSoup(contents, "html.parser")
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0
        all_images = soup.find_all("image")

        widgets = ["Converting ", dType, " ",progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(all_images), widgets=widgets).start()

        # Loop over all images
        for i, image in enumerate(all_images):
            # Load the image as a TensorFlow object
            p = os.path.sep.join([settings.PATH_BASE, image["file"]])
            encoded = tf.gfile.GFile(p, "rb").read()
            encoded = bytes(encoded)

            # Load the image again, now as a PIL object
            pilImage = Image.open(p)
            (w, h) = pilImage.size[:2]

            # Get the filename and encoding
            filename = image["file"].split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]

            # Create the data point
            tf_data_point = TFDataPoint()
            tf_data_point.image = encoded
            tf_data_point.encoding = encoding
            tf_data_point.filename = filename
            tf_data_point.width = w
            tf_data_point.height = h

            # Loop over all bounding boxes associated with the image
            for box in image.find_all("box"):
                # Skip boxes that can be ignored
                if box.has_attr("ignore"):
                    continue

                # Avoid going outside the image's dimensions
                startX = max(0, float(box["left"]))
                startY = max(0, float(box["top"]))
                endX = min(w, float(box["width"]) + startX)
                endY = min(h, float(box["height"]) + startY)
                label = box.find("label").text                  # label = class name (a string, not an int)

                # Scale to [0, 1] as required by TF
                xMin = startX/w
                xMax = endX/w
                yMin = startY/h
                yMax = endY/h

                # Check that box coordinates are valid, skip the box if they are not
                if xMin > xMax or yMin > yMax:
                    continue
                elif xMax < xMin or yMax < yMin:
                    continue

                # Add the box to the data point
                tf_data_point.xMins.append(xMin)
                tf_data_point.xMaxs.append(xMax)
                tf_data_point.yMins.append(yMin)
                tf_data_point.yMaxs.append(yMax)
                tf_data_point.classLabels.append(label.encode("utf8"))
                tf_data_point.classes.append(settings.CLASSES[label])
                tf_data_point.difficult.append(0)

                total += 1

            # Create the data point for the current image
            features = tf.train.Features(feature=tf_data_point.create_data_point())
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

            pbar.update(i)

        writer.close()
        pbar.finish()


if __name__ == "__main__":
    tf.app.run()
