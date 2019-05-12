import argparse
import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, \
preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, \
preprocess_true_boxes, yolo_loss, yolo_body

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
	## box_confidence (19, 19, 5, 1)
	## box_class_probs (19, 19, 5, 80) 80 classes
	box_scores = box_confidence * box_class_probs ## (19, 19, 5, 80)
	box_classes = K.argmax(box_scores, axis = -1) ## 
	box_class_scores = K.max(box_scores, axis = -1, keepdims = False)
	filtering_mask = box_class_scores >= threshold

	scores = tf.boolean_mask(box_class_scores, filtering_mask)
	boxes = tf.boolean_mask(boxes, filtering_mask)
	classes = tf.boolean_mask(box_classes, filtering_mask)	

	return scores, boxes, classes

# with tf.Session() as test_a:
# 	box_confidence = tf.random_normal([19, 19, 5, 1], mean = 1, stddev = 4,\
# 		seed = 1)
# 	boxes = tf.random_normal([19, 19, 5, 4], mean = 1, stddev = 4, seed = 1)
# 	box_class_probs = tf.random_normal([19, 19, 5, 80], mean = 1, stddev = 4 \
# 		, seed = 1)
# 	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, \
# 		box_class_probs, .5)

# 	print("scores[2] = " + str(scores[2].eval()))
# 	print("boxes[2] = " + str(boxes[2].eval()))
# 	print("classes[2] = " + str(classes[2].eval()))
# 	print("scores.shape = " + str(scores.shape))
# 	print("boxes.shape = " + str(boxes.shape))
# 	print("classes.shape = " + str(classes.shape))
	

def iou(box1, box2):

	xi1 = max(box1[0], box2[0])
	yi1 = max(box1[1], box2[1])
	xi2 = min(box1[2], box2[2])
	yi2 = min(box1[3], box2[3])
	inter_area = (yi2 - yi1) * (xi2 - xi1)

	box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
	box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
	union_area = box1_area + box2_area - inter_area

	iou = inter_area / union_area

	return iou

# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3, 4) 
# print("iou = " + str(iou(box1, box2)))

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10,  \
	iou_threshold = 0.5):
	# tensor to be used in tf.image.non_max_suppression()
	max_boxes_tensor = K.variable(max_boxes, dtype = 'int32')
	K.get_session().run(tf.variables_initializer(max_boxes_tensor))
	nms_indices = tf.image.non_max_suppression(boxes, scores, \
		max_boxes, iou_threshold)
	scores = K.gather(scores, nms_indices)
	boxes = K.gather(boxes, nms_indices)
	classes = K.gather(classes, nms_indices)

	return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes = 10, \
	score_threshold = .6, iou_threshold = .5):
	
	box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
	boxes = yolo_boxes_to_corners(box_xy, box_wh)

	# Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
	scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

	# Scale boxes back to original image shape.
	boxes = scale_boxes(boxes, image_shape)

	# Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
	scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

	return scores, boxes, classes

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)

yolo_model = load_model("model_data/yolo.h5")

yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

def predict(sess, image_file):
	image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
	out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], \
		feed_dict = {yolo_model.input:image_data, K.learning_phase():0})


