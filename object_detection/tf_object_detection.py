import os
import sys
import cv2
import time
import shutil
import argparse
import numpy as np
import tensorflow as tf
import multiprocessing
from natsort import natsorted
#from matplotlib import pyplot as plt

#sys.path.insert(0, "/home/berker/tensorflow/models/research")
sys.path += [sys.path.pop(0)]
#print "\n".join(sys.path)
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = "/".join(os.path.realpath(__file__).split('/')[:-1])

#faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28
#faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28
MODEL_NAME = "faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28"
PATH_TO_CKPT = os.path.join(CWD_PATH, "models", MODEL_NAME, "frozen_inference_graph.pb")
#PATH_TO_CKPT = os.path.join(CWD_PATH, "models", MODEL_NAME, "saved_model", "saved_model.pb")
PATH_TO_LABELS = os.path.join(CWD_PATH, "data", "oid_bbox_trainable_label_map.pbtxt")

NUM_CLASSES = 546

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

	
def detect_objects(image_np, sess, detection_graph, fp):
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Each box represents a part of the image where a particular object was detected.
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# Actual detection.
	(boxes, scores, classes, num_detections) = sess.run(
		[boxes, scores, classes, num_detections],
		feed_dict={image_tensor: image_np_expanded})
	"""
	print "boxes:", boxes
	print "scores:", scores
	print "classes:", classes
	print "num_detections:", num_detections, num_detections[0], "\n" """

	top_n = 3
	threshold = 0.5
	for index, value in enumerate(classes[0]):
		if index >= num_detections[0]:
			break
		print "\t", (category_index.get(value)).get('name').encode('utf8'), scores[0, index]
		fp.write('\t{} {}\n'.format((category_index.get(value)).get('name').encode('utf8'), scores[0, index]))
	print "\n"
	# TODO: implement better printing method

	"""
	for num in range(num_detections[0]):
		print [category_index.get(value) for index,value in enumerate(classes[num])]
	
	threshold = 0.01
	objects = []
	for index, value in enumerate(classes[0]):
		object_dict = {}
		#if scores[0, index] > threshold:
		object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
							scores[0, index]
		objects.append(object_dict)
	print objects
	print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > threshold])
	"""
	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=threshold)
	return image_np

# First test on images
PATH_TO_IMAGES_DIR = os.path.join(CWD_PATH, "input_images")
PATH_TO_OUTPUT_DIR = os.path.join(CWD_PATH, "output_images", MODEL_NAME)
if os.path.isdir(PATH_TO_OUTPUT_DIR):
	shutil.rmtree(PATH_TO_OUTPUT_DIR)
os.mkdir(PATH_TO_OUTPUT_DIR)
PATH_TO_OUTPUT = os.path.join(CWD_PATH, "output")

IMAGE_PATHS = []
for (dirpath, dirnames, filenames) in os.walk(PATH_TO_IMAGES_DIR):
	IMAGE_PATHS.extend(natsorted(filenames))
	break
	
"""
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)"""

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


from PIL import Image
"""for image_path in IMAGE_PATHS:
	image = Image.open(image_path)
	image_np = load_image_into_numpy_array(image)
	plt.imshow(image_np)
	print(image.size, image_np.shape)"""

#Load a frozen TF model
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		with open(os.path.join(PATH_TO_OUTPUT, MODEL_NAME + ".txt"), 'w') as fp:
			for image_path in IMAGE_PATHS:
				image = Image.open(os.path.join(PATH_TO_IMAGES_DIR, image_path))
				image_np = load_image_into_numpy_array(image)
				print image_path.split("/")[-1] + ":"
				fp.write(image_path.split("/")[-1] + ":\n")
				image_process = detect_objects(image_np, sess, detection_graph, fp)
				fp.write("\n")
				image_process = Image.fromarray(image_process, "RGB")
				image_process.save(os.path.join(PATH_TO_OUTPUT_DIR, image_path))
