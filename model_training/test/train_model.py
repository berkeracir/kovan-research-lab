import os
from natsort import natsorted

import tensorflow as tf
from object_detection.utils import dataset_util

ANNOTATION_DIR = "Annotation"
IMAGES_DIR = "Images"
RECORDS_DIR = "tfrecords"
LABEL_PATH = "tools_label_map.pbtxt"

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example():
	# TODO(user): Populate the following variables from your example.
	height = 500 # Image height
	width = 375 # Image width
	filename = "n04154565_323" # Filename of the image. Empty if image is not from file
	#encoded_image_data = None # Encoded image bytes
	image_format = b'jpg' # b'jpeg' or b'png'

	xmins = [22.0/375.0, 0.0/375.0] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [280.0/375.0, 499.0/375.0] # List of normalized right x coordinates in bounding box
				# (1 per box)
	ymins = [50.0/500.0, 104.0/500.0] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [210.0/500.0, 352.0/500.0] # List of normalized bottom y coordinates in bounding box
				# (1 per box)
	classes_text = ["n04154565", "n04154565"] # List of string class name of bounding box (1 per box)
	classes = [1, 1] # List of integer class id of bounding box (1 per box)

	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename),
		'image/source_id': dataset_util.bytes_feature(filename),
		#'image/encoded': dataset_util.bytes_feature(encoded_image_data),
		#'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	return tf_example


def main(_):
	if os.path.exists(LABEL_PATH):
		os.remove(LABEL_PATH)
	id = 1

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
				print image_id
		
		id += 1

	exit()

	writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

	tf_example = create_tf_example()
	writer.write(tf_example.SerializeToString())

	writer.close()

if __name__ == '__main__':
	tf.app.run()