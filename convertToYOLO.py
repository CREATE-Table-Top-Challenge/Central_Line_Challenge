"""Prepares the dataset for use in YOLOv5. Assumes training data is already
loaded into one folder."""

import os
import argparse
import glob
import shutil
import pandas as pd
from ruamel.yaml import YAML
from PIL import Image

FLAGS = None


def createDatasetDF(dataDirectory):
    videoIDs = [x for x in os.listdir(dataDirectory) if not "." in x]
    dfInitialized = False
    for video in videoIDs:
        labelFile = pd.read_csv(os.path.join(dataDirectory, video, video + "_Labels.csv"))
        labelFile["Folder"] = [os.path.join(dataDirectory, video) for i in range(len(labelFile.index))]
        if not dfInitialized:
            df = pd.DataFrame(columns=labelFile.columns)
            dfInitialized = True
        df = df.append(labelFile, ignore_index=True)
    for column in df.columns:
        if "Unnamed" in column:
            df = df.drop(column, axis=1)
    return df


def createDFAndYAML(yamlFilename, rootDir, trainDir, testDir=None):
    trainDF = createDatasetDF(os.path.join(rootDir, trainDir))
    trainFilenames = trainDF["FileName"].tolist()
    trainFilenames = ["images/" + name for name in trainFilenames]
    testDF = None
    if testDir:
        testDF = createDatasetDF(os.path.join(rootDir, testDir))
        testFilenames = testDF["FileName"].tolist()
        testFilenames = [name[:-9] + "/" + name for name in testFilenames]
    labels = trainDF["Tool"].unique().tolist()
    data = {
        "path": rootDir,
        "train": trainFilenames,
        "val": trainFilenames,
        "test": testFilenames if testDir else None,
        "nc": len(labels),
        "names": labels
    }
    with open(yamlFilename, "w") as f:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(data, f)
    print(f"Created {yamlFilename} file in YOLO format.")
    return trainDF, testDF, labels


def reorganizeDirectories(rootDir, trainDir, newRelativeDir):
    destination = os.path.join(rootDir, newRelativeDir)
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.mkdir(destination)
    pattern = os.path.join(rootDir, trainDir) + "/*/*"
    for img in glob.glob(pattern):
        shutil.copy(img, destination)  # maybe change to move?


def createLabels(df, labels, rootDir):
    labelDir = os.path.join(rootDir, "labels")
    if os.path.exists(labelDir):
        shutil.rmtree(labelDir)
    os.mkdir(labelDir)
    df = df.reset_index()  # ensure indexes are in order
    for idx, row in df.iterrows():
        im = Image.open(os.path.join(rootDir, "images", row["FileName"]))
        im_width, im_height = im.size
        labelFilename = row["FileName"][:-4] + ".txt"
        labelPath = os.path.join(labelDir, labelFilename)
        data = ""
        with open(labelPath, "w") as f:
            objects = list(eval(row["Tool bounding box"]))
            for obj in objects:
                labelIndex = labels.index(obj["class"])
                width = obj["xmax"] - obj["xmin"]
                height = obj["ymax"] - obj["ymin"]
                xCenter = (obj["xmin"] + obj["xmax"]) / 2
                yCenter = (obj["ymin"] + obj["ymax"]) / 2
                # Normalize coordinates
                xCenter /= im_width
                width /= im_width
                yCenter /= im_height
                height /= im_height
                data += f"{labelIndex} {xCenter} {yCenter} {width} {height}\n"
            f.write(data)


def createDataset():
    if FLAGS.data_location == "":
        print("No data specified. Please set flag --data_location")
    elif FLAGS.train_data_relative_path == "":
        print("No training data specified. Please set flag --train_data_relative_path")
    else:
        basePath = FLAGS.data_location
        trainLocation = FLAGS.train_data_relative_path
        testLocation = None
        if FLAGS.test_data_relative_path != "":
            testLocation = FLAGS.test_data_relative_path
        trainDF, testDF, labels = createDFAndYAML(
            FLAGS.yaml_filename, basePath, trainLocation, testLocation
        )
        reorganizeDirectories(basePath, trainLocation, "images")
        labelPath = os.path.join(basePath, "labels")
        createLabels(trainDF, labels, basePath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data_location",
      type=str,
      default="",
      help="Name of the directory where the training/testing data is located"
  )
  parser.add_argument(
      "--train_data_relative_path",
      type=str,
      default="",
      help="Relative path from 'data_location' to the directory where the training data is located"
  )
  parser.add_argument(
      "--test_data_relative_path",
      type=str,
      default="",
      help="Relative path from 'data_location' to the directory where the "
           "testing data is located, if it exists"
  )
  parser.add_argument(
      "--yaml_filename",
      type=str,
      default="",
      help="Name of the yaml file to which to write data in YOLO format"
  )

  FLAGS, unparsed = parser.parse_known_args()
  createDataset()
