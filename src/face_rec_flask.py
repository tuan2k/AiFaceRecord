from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import render_template, flash, redirect, url_for, request
from flask_cors import CORS, cross_origin
import tensorflow as tf
from werkzeug.utils import secure_filename
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
import base64

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
facenet.load_model(FACENET_MODEL_PATH)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# Get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Dataset/FaceData/TFace.db'
app.config['SECRET_KEY'] = 'Edward secret key <3'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CORS(app)



@app.route('/')
@cross_origin()
def index():
    return "OK!";

@app.route('/recog', methods=['POST'])
@cross_origin()
def upload_img_file():
    if request.method == 'POST':
        # base 64
        name="Unknown"

        f = request.files.get('image')

        image = np.asarray(bytearray(f.read()), dtype="uint8")

        frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

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
                    scaled = facenet.prewhiten(scaled)
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
                    if best_class_probabilities > 0.8:
                        name = class_names[best_class_indices[0]]
                    else:
                        # Con neu <=0.5 thi hien thi Unknow
                        name = "Unknown"
        except:
            pass
    return name;

@app.route('/tranningAgain', methods=['GET'])
@cross_origin()
def align_dataset_mtcnn_again():
    os.system("python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25")
    os.system("python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000")
    return "Success"

@app.route('/tranning', methods=['POST'])
@cross_origin()
def align_dataset_mtcnn():
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)

    files = request.files.getlist('files')
    rs_username = request.form['txtusername']
    directory = rs_username.replace(" ","")
    parent_dir = "C:/Users/Admin/PycharmProjects/TFaceV1Project/Dataset/FaceData/raw/"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)

    print("Directory '% s' created" % directory)

    UPLOAD_FOLDER = 'C:/Users/Admin/PycharmProjects/TFaceV1Project/Dataset/FaceData/raw'\
                    + '/' + directory
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    for file in files:
        filename = secure_filename(file.filename)

        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File successfully uploaded ' + file.filename + ' to the database!')

    os.system("python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25")
    os.system("python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000")

    return "Success"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port='8000')

