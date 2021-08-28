from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import shutil
from os import listdir,path,walk
from os.path import isfile, join
from random import random

import flask
import requests

import urllib.request

from PIL import Image, ImageEnhance
from flask import Flask, make_response, jsonify
from flask import render_template, flash, redirect, url_for, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from werkzeug.utils import secure_filename
import src.facenet
import os

import pickle
import src.align.detect_face
import numpy as np
import cv2


MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
IMAGE_SIZE = 182
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'Models/facemodel.pkl'
FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

tf.Graph().as_default()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))


# Load the model
print('Loading feature extraction model')
src.facenet.load_model(FACENET_MODEL_PATH)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER_RAW = os.path.expanduser("Dataset/FaceData/raw")

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, "src/align")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Dataset/FaceData/TFace.db'
app.config['UPLOAD_FOLDER_RAW'] = UPLOAD_FOLDER_RAW
app.config['SECRET_KEY'] = 'Edward secret key <3'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app)



@app.route('/')
@cross_origin()
def index():
    return "OK!";

@app.route('/identified', methods=['POST'])
@cross_origin()
def upload_img_file():
    if request.method == 'POST':
        # base 64
        name="Unknown"

        f = request.files.get('image')

        image = np.asarray(bytearray(f.read()), dtype="uint8")

        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        bounding_boxes, _ = src.align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]
        try:
            # Neu co it nhat 1 khuon mat trong frame
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # Cat phan khuon mat tim duoc
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = src.facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    # Dua vao model de classifier
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]

                    # Lay ra ten va ty le % cua class co ty le cao nhat
                    best_name = class_names[best_class_indices[0]]
                    print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                    # Neu ty le nhan dang > 0.5 thi hien thi ten
                    if best_class_probabilities > 0.7:
                        name = class_names[best_class_indices[0]]
                    else:
                        # Con neu <=0.5 thi hien thi Unknow
                        name = "Unknown"
        except:
            pass
    return name;

@app.route('/identifiedStr', methods=['POST'])
@cross_origin()
def upload_img_file_str():
    if request.method == 'POST':
        # base 64
        name="NULL"

        url_img = request.json['url']

        f = requests.get(url_img, stream=True)

        image = np.asarray(bytearray(f.content), dtype="uint8")

        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        bounding_boxes, _ = src.align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

        faces_found = bounding_boxes.shape[0]

        try:
            # Neu co it nhat 1 khuon mat trong frame
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # Cat phan khuon mat tim duoc
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = src.facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)

                    # Dua vao model de classifier
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]

                    # Lay ra ten va ty le % cua class co ty le cao nhat
                    best_name = class_names[best_class_indices[0]]
                    print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                    # Neu ty le nhan dang > 0.5 thi hien thi ten
                    if best_class_probabilities > 0.9:
                        name = class_names[best_class_indices[0]]
                    else:
                        # Con neu <=0.5 thi hien thi Unknow
                        return flask.jsonify()
        except:
            pass
    return flask.jsonify(name_rp=name)

@app.route('/identifiedStrListTest', methods=['POST'])
@cross_origin()
def upload_img_file_str_list_test():
    mypath = os.path.expanduser("Dataset/FaceData/processed")
    onlyfiles = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
    list = [onlyfiles[np.random.randint(0, len(onlyfiles))],onlyfiles[np.random.randint(0, len(onlyfiles))]
            ,onlyfiles[np.random.randint(0, len(onlyfiles))]];
    return jsonify(name_list_rp=list)

@app.route('/identifiedStrList', methods=['POST'])
@cross_origin()
def upload_img_file_str_list():
    if request.method == 'POST':
        # base 64
        name="Unknown"

        name_list = []

        url_img_list = request.json['url_list']

        for url_img in url_img_list:

            f = requests.get(url_img, stream=True)

            image = np.asarray(bytearray(f.content), dtype="uint8")

            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            bounding_boxes, _ = src.align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

            faces_found = bounding_boxes.shape[0]

            try:
                # Neu co it nhat 1 khuon mat trong frame
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # Cat phan khuon mat tim duoc
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = src.facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                        # Dua vao model de classifier
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]

                        # Lay ra ten va ty le % cua class co ty le cao nhat
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                        # Neu ty le nhan dang > 0.5 thi hien thi ten
                        if best_class_probabilities > 0.9:
                            name_list.append(class_names[best_class_indices[0]])
                            # Con neu <=0.5 thi hien thi Unknow

            except:
                pass
    return flask.jsonify(name_list_rp=name_list)

@app.route('/trainning', methods=['GET'])
@cross_origin()
def align_dataset_mtcnn_again():
    os.system("python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000")
    return make_response(
            jsonify(
                {"message": "Success"}
            ),
            200,
        )

@app.route('/tranningTest', methods=['GET'])
@cross_origin()
def align_dataset_mtcnn_again_test():
    return make_response(
        jsonify(
            {"message": "Success"}
        ),
        200,
    )

@app.route('/checkAvailable', methods=['GET'])
@cross_origin()
def check_available():
    return make_response(
        jsonify(
            {"message": "Success"}
        ),
        200,
    )

@app.route('/deleteByIdTest', methods=['POST'])
@cross_origin()
def deleteByIdTest():
    if not request.json['id']:
        return make_response(
            jsonify(
                {"message": "Error"}
            ),
            400,
        )
    else:
        return make_response(
            jsonify(
                {"message": "Delete success ID: " + str(id)}
            ),
            200,
        )

@app.route('/deleteById', methods=['POST'])
@cross_origin()
def deleteById():
    id = request.json['id']

    if not id:
        return make_response(
            jsonify(
                {"message": "Error"}
            ),
            400,
        )
    parent_dir = os.path.expanduser("Dataset/FaceData/raw")
    path = os.path.join(parent_dir, id)
    shutil.rmtree(path, ignore_errors=True)
    parent_dir = os.path.expanduser("Dataset/FaceData/processed")
    path = os.path.join(parent_dir, id)
    shutil.rmtree(path, ignore_errors=True)

    return make_response(
            jsonify(
                {"message": "Success"}
            ),
            200,
        )

@app.route('/uploadByUrlsTest', methods=['POST'])
@cross_origin()
def align_dataset_mtcnn_url_test():
    url_img_list = request.json['url_list']

    if len(url_img_list) != 20:
        return make_response(
            jsonify(
                {"message": "Need 20 images for tranning"}
            ),
            400,
        )
    else:
        return make_response(
                jsonify(
                    {"message": "Success"}
                ),
                200,
            )


@app.route('/uploadByUrls', methods=['POST'])
@cross_origin()
def align_dataset_mtcnn_url():
    rs_username = request.json['txtusername']
    directory = rs_username.replace(" ", "")
    parent_dir = os.path.expanduser("Dataset/FaceData/raw")
    path = os.path.join(parent_dir, directory)

    try:
        os.mkdir(path)
        print("Directory '% s' created" % directory)
    except OSError as err:
        message = "OS error: {0}".format(err)
        return make_response(
            jsonify(
                {"message": message}
            ),
            400,
        )
    url_img_list = request.json['url_list']

    if len(url_img_list) != 20:
        return make_response(
            jsonify(
                {"message": "Need 20 images for tranning"}
            ),
            400,
        )

    UPLOAD_FOLDER = os.path.expanduser("Dataset/FaceData/raw") + '/' + directory
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    for i, url_img in enumerate(url_img_list):
        filename = 'image' + str(i) + '.jpg'
        urllib.request.urlretrieve(url_img, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfulliy uploaded ' + filename + ' to the database!')

    os.system("python src/align_dataset_mtcnn.py  Dataset/FaceData/raw"
              + " Dataset/FaceData/processed --image_size 160 "
              + "--margin 32  --random_order --gpu_memory_fraction 0.25")

    output_dir = os.path.expanduser("Dataset/FaceData/processed/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nrof_images_total = 0
    output_class_dir = os.path.join(output_dir, directory)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    for image_path in src.facenet.get_imgDirs("Dataset/FaceData/processed/" + directory):
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0] + str(nrof_images_total)
        output_filename = os.path.join(output_class_dir, filename + '.png')
        print(image_path)
        image_obj = Image.open(image_path);
        rotated_image2 = image_obj.convert('L')
        rotated_image2.save(output_filename)
        image_flip = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        output_filename = os.path.join(output_class_dir, filename + 'a.png')
        image_flip.save(output_filename);
        output_filename = os.path.join(output_class_dir, filename + 'b.png')
        sharpness = ImageEnhance.Sharpness(image_obj);
        sharpness.enhance(1.5).save(output_filename);
        output_filename = os.path.join(output_class_dir, filename + 'c.png')
        color = ImageEnhance.Color(image_obj)
        color.enhance(1.5).save(output_filename)
        output_filename = os.path.join(output_class_dir, filename + 'd.png')
        brightness = ImageEnhance.Brightness(image_obj)
        brightness.enhance(1.5).save(output_filename);

    return make_response(
        jsonify(
            {"message": "Success"}
        ),
        200,
    )

@app.route('/upload', methods=['POST'])
@cross_origin()
def align_dataset_mtcnn():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)

    files = request.files.getlist('files')
    rs_username = request.form['txtusername']
    directory = rs_username.replace(" ","")
    parent_dir = os.path.expanduser("Dataset/FaceData/raw")
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    print("Directory '% s' created" % directory)

    UPLOAD_FOLDER = os.path.expanduser("Dataset/FaceData/raw") + '/' + directory
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    for file in files:
        filename = secure_filename(file.filename)

        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded ' + file.filename + ' to the database!')

    print(files)

    os.system("python src/align_dataset_mtcnn.py  Dataset/FaceData/raw"
              + " Dataset/FaceData/processed --image_size 160 "
              + "--margin 32  --random_order --gpu_memory_fraction 0.25")

    output_dir = os.path.expanduser("Dataset/FaceData/processed/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nrof_images_total = 0
    output_class_dir = os.path.join(output_dir, directory)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)
    for image_path in src.facenet.get_imgDirs("Dataset/FaceData/processed/" + directory):
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0] + str(nrof_images_total)
        output_filename = os.path.join(output_class_dir, filename + '.png')
        print(image_path)
        image_obj = Image.open(image_path);
        rotated_image2 = image_obj.convert('L')
        rotated_image2.save(output_filename)
        image_flip = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        output_filename = os.path.join(output_class_dir, filename + '2.png')
        image_flip.save(output_filename);
        output_filename = os.path.join(output_class_dir, filename + '1.png')
        sharpness = ImageEnhance.Sharpness(image_obj);
        sharpness.enhance(1.5).save(output_filename);
        output_filename = os.path.join(output_class_dir, filename + '3.png')
        color = ImageEnhance.Color(image_obj)
        color.enhance(1.5).save(output_filename)
        output_filename = os.path.join(output_class_dir, filename + '4.png')
        brightness = ImageEnhance.Brightness(image_obj)
    return "Success"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    extra_dirs = ['./Models/', ]
    extra_files = extra_dirs[:]
    for extra_dir in extra_dirs:
        for dirname, dirs, files in walk(extra_dir):
            for filename in files:
                filename = path.join(dirname, filename)
                if path.isfile(filename):
                    extra_files.append(filename)
    app.run(extra_files=extra_files,debug=True, host='0.0.0.0',port='8000')
