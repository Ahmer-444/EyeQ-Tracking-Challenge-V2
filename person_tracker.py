#!/usr/bin/python
# coding: utf-8
import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
print('Using Tensorflow Version ' + str(tf.__version__))
import zipfile
import re
import glob
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time

sys.path.append("..")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from nms import non_max_suppression_fast

from centroid_tracker.centroidtracker import CentroidTracker
from centroid_tracker.trackableobject import TrackableObject

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='input_video_path',
                    help='video to be processed')
parser.add_argument('-o', dest='vid_record_path',
                    help='recorded video path')
args = parser.parse_args()

INPUT_VIDEO_PATH =  args.input_video_path
VID_RECORD_PATH = args.vid_record_path

print ('input_video_path     =', INPUT_VIDEO_PATH)
print ('recorded_video_path     =', VID_RECORD_PATH)

write_fname = 'predictions.txt'
exists = os.path.isfile(write_fname)
if exists:
  os.remove(write_fname)

PERSON_TH = 0.80

# Gloabl Variables
person_image_tensor = None
person_detection_boxes = None
person_detection_scores = None
person_detection_classes = None
person_num_detections = None

ct = CentroidTracker(maxDisappeared=100, maxDistance=150)
trackableObjects = {}



def VideoSrcInit():
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    #cap = skvideo.io.vread(INPUT_VIDEO_PATH)
    flag, image = cap.read()
    if flag:
        print("Valid Video Path. Lets move to detection!")
    else:
        raise ValueError("Video Initialization Failed. Please make sure video path is valid.")
    return cap

def VideoRecInit(WIDTH,HEIGHT):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(VID_RECORD_PATH, fourcc, 30.0, (WIDTH,HEIGHT))
    return videowriter


cap = VideoSrcInit()
flag, image = cap.read()
(ht,wd,_) = image.shape
videowriter = VideoRecInit(wd,ht)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PERSON_PATH_TO_CKPT = '/home/sami_malik914/tensorflow_src/pretrained_models/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
S_PATH_TO_LABELS = 'inputs/person_label_map.pbtxt'

NUM_CLASSES = 1


# ## Load a (frozen) Tensorflow model into memory.
person_detection_graph = tf.Graph()
with person_detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PERSON_PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



# ## Loading hole label map
s_label_map = label_map_util.load_labelmap(S_PATH_TO_LABELS)
s_categories = label_map_util.convert_label_map_to_categories(s_label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
s_category_index = label_map_util.create_category_index(s_categories)



with person_detection_graph.as_default():
  with tf.Session(graph=person_detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    person_image_tensor = person_detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    person_detection_boxes = person_detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    person_detection_scores = person_detection_graph.get_tensor_by_name('detection_scores:0')
    person_detection_classes = person_detection_graph.get_tensor_by_name('detection_classes:0')
    person_num_detections = person_detection_graph.get_tensor_by_name('num_detections:0')
    counter = 0
    fps_list = []
    frame_no = 0
    skip_frames = 0
    while True:
      frame_no += 1
      start_time = time.time()
      print ('frame_no: ' + str(frame_no))
      #if frame_no < 5300:
      #  continue
      flag, image = cap.read()
      if flag == False:
        break

      #if frame_no % skip_frames != 0:
      #  continue

      image_np = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      fetch_time = time.time()

      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
      	[person_detection_boxes, person_detection_scores, person_detection_classes, person_num_detections],
      	feed_dict={person_image_tensor: image_np_expanded})


      sboxes = np.squeeze(boxes);
      sclasses = np.squeeze(classes).astype(np.int32);
      sscores = np.squeeze(scores);

      frame,bboxes = vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              sboxes,
              sclasses,
              sscores,
              s_category_index,
              min_score_thresh=PERSON_TH,
              max_boxes_to_draw=7,
              use_normalized_coordinates=True,
	      skip_scores=True,
              line_thickness=8)
      #xmin,ymin,xmax,ymax = bboxes
      #rects.append((xmin,ymin,xmax,ymax))
      rects = non_max_suppression_fast(bboxes,0.80)
      objects = ct.update(rects)

      # loop over the tracked objects
      for (objectID, centroid) in objects.items():
            print(objects.keys())
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                y = [c[1] for c in to.centroids]
                #direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # if not CentroidTracker.FLAG:
            #print(CentroidTracker.FLAG)
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

      videowriter.write(frame)
      fps = 1.0 / (time.time() - start_time)
      print("--- %s FPS ---" % fps)

    videowriter.release()
    cap.release()

