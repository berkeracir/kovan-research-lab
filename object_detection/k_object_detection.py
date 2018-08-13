import os
import sys
import shutil
from natsort import natsorted

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

# inserting data/ssd_keras module into PYTHONPATH
CWD_PATH = "/".join(os.path.realpath(__file__).split('/')[:-1])
DATA_PATH = os.path.join(CWD_PATH, "data", "ssd_keras")
sys.path.insert(0, DATA_PATH)

from models.keras_ssd300 import ssd_300
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 300
img_width = 300

# Build the model and load trained weights into it
K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=200,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], #[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

weights_path = os.path.join(CWD_PATH, "weights", "VGG_ILSVRC2016_SSD_300x300_iter_440000.h5")
model_name = "VGG_ILSVRC2016_SSD_300x300_iter_440000"

model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

"""
# Load a trained model
model_path = os.path.join(CWD_PATH, "models", "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5")

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
											   'L2Normalization': L2Normalization,
											   'DecodeDetections': DecodeDetections,
											   'compute_loss': ssd_loss.compute_loss})
"""


PATH_TO_IMAGES_DIR = os.path.join(CWD_PATH, "input_images")
PATH_TO_OUTPUT_DIR = os.path.join(CWD_PATH, "output_images", model_name)
if os.path.isdir(PATH_TO_OUTPUT_DIR):
	shutil.rmtree(PATH_TO_OUTPUT_DIR)
os.mkdir(PATH_TO_OUTPUT_DIR)
PATH_TO_OUTPUT = os.path.join(CWD_PATH, "output")

IMAGE_PATHS = []
for (dirpath, dirnames, filenames) in os.walk(PATH_TO_IMAGES_DIR):
	IMAGE_PATHS.extend(natsorted(filenames))
	break

with open(os.path.join(PATH_TO_OUTPUT, model_name + ".txt"), 'w') as fp:
	for image_path in IMAGE_PATHS:
		orig_image = imread(os.path.join(PATH_TO_IMAGES_DIR, image_path))
		img = image.load_img(os.path.join(PATH_TO_IMAGES_DIR, image_path), target_size=(img_height, img_width))
		img = image.img_to_array(img)

		y_pred = model.predict(np.array([img]))
		confidence_threshold = 0.5

		y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
		print(image_path.split("/")[-1] + ":")
		print(y_pred_thresh[0])
		print("\n")
		fp.write(image_path.split("/")[-1] + ":\n")
		fp.write("\t")
		fp.write(y_pred_thresh[0])
		fp.write("\n")

		colors = plt.cm.hsv(np.linspace(0, 1, 201)).tolist()
		# TODO: implement class names
		classes = ['background',
				'aeroplane', 'bicycle', 'bird', 'boat',
				'bottle', 'bus', 'car', 'cat',
				'chair', 'cow', 'diningtable', 'dog',
				'horse', 'motorbike', 'person', 'pottedplant',
				'sheep', 'sofa', 'train', 'tvmonitor']

		#plt.figure(figsize=(20,12))
		plt.imshow(imread(os.path.join(PATH_TO_IMAGES_DIR, image_path)))

		current_axis = plt.gca()

		for box in y_pred_thresh[0]:
			# Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
			
			xmin = box[2] * orig_image.shape[1] / img_width
			ymin = box[3] * orig_image.shape[0] / img_height
			xmax = box[4] * orig_image.shape[1] / img_width
			ymax = box[5] * orig_image.shape[0] / img_height
			color = colors[int(box[0])]
			#label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
			label = '{}: {:.2f}'.format(int(box[0]), box[1])
			current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
			current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

		plt.savefig(os.path.join(PATH_TO_OUTPUT_DIR, image_path.split("/")[-1]))
		plt.close()
		
"""
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = os.path.join(DATA_PATH,'examples', 'fish_bike.jpg')

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img)
input_images.append(img)
input_images = np.array(input_images)

y_pred = model.predict(input_images)
confidence_threshold = 0.5

y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

# Display the image and draw the predicted boxes onto it.

# Set the colors for the bounding boxes
colors = plt.cm.hsv(np.linspace(0, 1, 201)).tolist()
classes = ['background',
		   'aeroplane', 'bicycle', 'bird', 'boat',
		   'bottle', 'bus', 'car', 'cat',
		   'chair', 'cow', 'diningtable', 'dog',
		   'horse', 'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor']

plt.figure(figsize=(20,12))
plt.imshow(imread(img_path))

current_axis = plt.gca()

for box in y_pred_thresh[0]:
	# Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
	xmin = box[2] * orig_images[0].shape[1] / img_width
	ymin = box[3] * orig_images[0].shape[0] / img_height
	xmax = box[4] * orig_images[0].shape[1] / img_width
	ymax = box[5] * orig_images[0].shape[0] / img_height
	color = colors[int(box[0])]
	#label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
	label = '{}: {:.2f}'.format(int(box[0]), box[1])
	current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
	current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

plt.show()"""