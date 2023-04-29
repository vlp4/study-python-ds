import numpy as np
import os
import re
import shutil
import sys
from pathlib import Path

from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection

SOURCE_DIR = 'data/chess_source'
DATA_DIR = 'data/chess'


def extract_filename_loss(fname):
    match = re.search('loss-(\\d+.\\d+).h5', str(fname))
    return match.group(1) if match else '?'


def find_best_model_file():
    models = sorted([i for i in Path(DATA_DIR + '/models').glob('*.h5')], key=extract_filename_loss, reverse=True)
    return str(models[-1]) if len(models) > 0 else None


def prepare_data():
    root_annots_path = SOURCE_DIR + '/annotations/'
    root_images_path = SOURCE_DIR + '/images/'

    annots_path = sorted([i for i in Path(root_annots_path).glob('*.xml')])
    images_path = sorted([i for i in Path(root_images_path).glob('*.png')])

    os.makedirs(DATA_DIR + '/train/images', exist_ok=True)
    os.makedirs(DATA_DIR + '/train/annotations', exist_ok=True)

    os.makedirs(DATA_DIR + '/validation/images', exist_ok=True)
    os.makedirs(DATA_DIR + '/validation/annotations', exist_ok=True)

    os.makedirs(DATA_DIR + '/test/images', exist_ok=True)
    os.makedirs(DATA_DIR + '/test/annotations', exist_ok=True)

    n_imgs = 81
    n_split = n_imgs // 6

    for i, (annot_path, img_path) in enumerate(zip(annots_path, images_path)):
        if i > n_imgs:
            break
        # train-val-test split
        if i < n_split:
            shutil.copy(img_path, DATA_DIR + '/test/images/' + img_path.parts[-1])
            shutil.copy(annot_path, DATA_DIR + '/test/annotations/' + annot_path.parts[-1])
        elif n_split <= i < n_split * 2:
            shutil.copy(img_path, DATA_DIR + '/validation/images/' + img_path.parts[-1])
            shutil.copy(annot_path, DATA_DIR + '/validation/annotations/' + annot_path.parts[-1])
        else:
            shutil.copy(img_path, DATA_DIR + '/train/images/' + img_path.parts[-1])
            shutil.copy(annot_path, DATA_DIR + '/train/annotations/' + annot_path.parts[-1])

    print(len(list(Path(DATA_DIR + '/train/annotations/').glob('*.xml'))))
    print(len(list(Path(DATA_DIR + '/validation/annotations/').glob('*.xml'))))
    print(len(list(Path(DATA_DIR + '/test/annotations/').glob('*.xml'))))


def train():
    prepare_data()

    best_model = find_best_model_file()
    print(f'Best model file is ', best_model)

    classes = np.array(["black-king", "white-king",
                        "black-pawn", "white-pawn",
                        "white-knight", "black-knight",
                        "black-bishop", "white-bishop",
                        "white-rook", "black-rook",
                        "black-queen", "white-queen"])

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=DATA_DIR + '/')
    trainer.setTrainConfig(object_names_array=classes,
                           batch_size=8,
                           num_experiments=30,
                           train_from_pretrained_model=best_model
                           )
    trainer.trainModel()


def detect():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    best_model = find_best_model_file()
    detector.setModelPath(best_model)
    detector.setJsonPath(DATA_DIR + "/json/detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(minimum_percentage_probability=60,
                                                 input_image=DATA_DIR + "/train/images/chess4.png",
                                                 output_image_path="detected.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


command = sys.argv[1]
if command == 'train':
    train()
elif command == 'detect':
    detect()
