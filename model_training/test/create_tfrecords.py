import os
import errno
import xml.etree.ElementTree as ET
from natsort import natsorted

import tensorflow as tf
from object_detection.utils import dataset_util

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

ANNOTATION_DIR = "Annotation"
IMAGES_DIR = "Images"
RECORDS_DIR = "tfrecords"
LABEL_PATH = "tools_label_map.pbtxt"

num_shards = 10
output_filebase = os.path.join(RECORDS_DIR, "tools_dataset.record")

"""flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS"""

def create_tf_example(wnid, image, id):
	image_id = image.split(".")[0]
	extension = image.split(".")[-1]
	image_path = os.path.join(IMAGES_DIR, wnid, image)
	image_xml = os.path.join(ANNOTATION_DIR, wnid, image_id + ".xml")
	root = ET.parse(image_xml).getroot()

	# TODO(user): Populate the following variables from your example.
	height = int(root[3][1].text) # Image height
	width = int(root[3][0].text) # Image width
	filename = image_path # Filename of the image. Empty if image is not from file
	image_format = bytes(extension) # b'jpeg' or b'png'

	xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [] # List of normalized right x coordinates in bounding box
				# (1 per box)
	ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [] # List of normalized bottom y coordinates in bounding box
				# (1 per box)
	classes_text = [] # List of string class name of bounding box (1 per box)
	classes = [] # List of integer class id of bounding box (1 per box)

	for xmin in root.iter('xmin'):
		xmins.append(float(xmin.text)/width)
	for xmax in root.iter('xmax'):
		xmaxs.append(float(xmax.text)/width)
	for ymin in root.iter('ymin'):
		ymins.append(float(ymin.text)/height)
	for ymax in root.iter('ymax'):
		ymaxs.append(float(ymax.text)/height)
	for name in root.iter('name'): # TODO: this may cause error in case of different bounding box objects
		classes_text.append(name.text)
		classes.append(id)

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename),
		'image/source_id': dataset_util.bytes_feature(filename),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example


def main(_):
	with contextlib2.ExitStack() as tf_record_close_stack:
		if not os.path.exists(os.path.dirname(output_filebase)):
			try:
				os.makedirs(os.path.dirname(output_filebase))
			except OSError as exc:
				if exc.errno != errno.EEXIST:
					raise
		output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_filebase, num_shards)

		if os.path.exists(LABEL_PATH):
			os.remove(LABEL_PATH)
		id = 1
		index = 0

		for wnid in natsorted(os.listdir(IMAGES_DIR)):
			# For each wnid, add a label
			with open(LABEL_PATH, 'a') as pbtxt:
				pbtxt.write("item {\n")
				pbtxt.write("  name: \"" + wnid + "\"\n")
				pbtxt.write("  id: " + str(id) + "\n")
				pbtxt.write("  display_name: \"" + str(id) + "\"\n")
				pbtxt.write("}\n")

			for image in natsorted(os.listdir(os.path.join(IMAGES_DIR, wnid))):
				image_id = image.split(".")[0]
				image_path = os.path.join(IMAGES_DIR, wnid, image)
				image_xml = os.path.join(ANNOTATION_DIR, wnid, image_id + ".xml")

				if os.path.isfile(image_xml):
					# Create necessary directories and files for tfrecords 
					tfrecords_path = os.path.join(RECORDS_DIR, wnid, image_id + ".tfrecords")
					if not os.path.exists(os.path.dirname(tfrecords_path)):
						try:
							os.makedirs(os.path.dirname(tfrecords_path))
						except OSError as exc:
							if exc.errno != errno.EEXIST:
								raise
					open(tfrecords_path, 'a').close()

					flags = tf.app.flags
					flags.DEFINE_string('output_path', tfrecords_path, 'Path to output TFRecord')
					FLAGS = flags.FLAGS
					
					writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

					tf_example = create_tf_example(wnid, image, id)
					
					writer.write(tf_example.SerializeToString())
					writer.close()

					output_shard_index = index % num_shards
					output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

					for name in list(FLAGS):
						delattr(FLAGS, name)

					index += 1
			
			id += 1

if __name__ == '__main__':
	tf.app.run()