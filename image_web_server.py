from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.backend import set_session
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer


UPLOAD_FOLDER = 'www/upload/'
RESULT_FOLDER = 'www/result'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'JPG', 'PNG', 'JPEG'])
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
COLORSET = {'aeroplane':'aqua', 'bicycle':'red', 'bird':'chocolate', 'boat':'darkkhaki',
           'bottle':'floralwhite', 'bus':'gray', 'car':'green', 'cat':'ivory', 'chair':'indigo',
           'cow':'blue', 'diningtable':'silver', 'dog':'lime', 'horse':'violet',
           'motorbike':'purple', 'person':'lightblue', 'pottedplant':'salmon',
           'sheep':'mintcream', 'sofa':'orange', 'train':'peru', 'tvmonitor':'pink'}
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def demo(sess, net, image_name, conter, result = True):
    im_read = cv2.imread(image_name, cv2.IMREAD_ANYCOLOR)
    b, g, r = cv2.split(im_read)
    im = cv2.merge([r, g, b])
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    fig, ax = plt.subplots(figsize=(12, 12))
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  #  skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            conter +=1
            continue
        ax.imshow(im, aspect='equal')
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=COLORSET[cls], linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(cls, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    if conter == len(CLASSES)-1:
        result = False
        print(result)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()
    return args

def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'r') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream

def load_model():
    global sess
    global net
    global graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    graph = tf.get_default_graph()
    set_session(sess)
    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', DATASETS[dataset][0], 'default', NETS[demonet][0])
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    else:
        raise NotImplementedError
    n_classes = len(CLASSES)
    # create the structure of the net having a certain shape (which depends on the number of classes)
    net.create_architecture(sess, "TEST", n_classes,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print('Loaded network {:s}'.format(tfmodel))

app = Flask(__name__)
load_model()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER



@app.route('/', methods=['GET'])
def send_index():
    return send_from_directory('www', "index.html")

@app.route('/www/<path:path>', methods=['GET'])
def send_root(path):
    return send_from_directory('www', path)

@app.route('/predict', methods=['POST'])
def upload_image1():
    # check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No posted image. Should be attribute named image.'})
    file = request.files['image']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return jsonify({'error': 'Empty filename submitted.'})
    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cachedir = os.path.join(app.config['RESULT_FOLDER'],filename)
        if os.path.exists(cachedir):
            response = {'url': "www/result/{:s}".format(filename.split('/')[-1])}
            print(response)
            return jsonify(response)
        else:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            conter = 0
            result = True
            result = demo(sess, net, filename, conter, result)
            if result == True:
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig("www/result/{:s}".format(filename.split('/')[-1]), bbox_inches='tight', transparent=True, pad_inches=0.0)
                response = {'url': "www/result/{:s}".format(filename.split('/')[-1])}
                print(response)
                return jsonify(response)
            else:
                response = {'false': "fail to classify"}
                print(response)
                return jsonify(response)
    else:
        return jsonify({'error':'File has invalid extension'})

if __name__ == '__main__':
    app.run(host= 'localhost',debug=True)