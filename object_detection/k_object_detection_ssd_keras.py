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
		fp.write(image_path.split("/")[-1] + ":\n")

		colors = plt.cm.hsv(np.linspace(0, 1, 201)).tolist()
		# TODO: implement class names
		classes = [ 'background', 
					"accordion", "airplane", "ant", "antelope", "apple", "armadillo", "artichoke", "axe", "baby bed", "backpack", 
					"bagel", "balance beam", "banana", "band aid", "banjo", "baseball", "basketball", "bathing cap", "beaker", 
					"bear", "bee", "bell pepper", "bench", "bicycle", "binder", "bird", "bookshelf", "bow", 
					"bow tie", "bowl", "brassiere", "burrito", "bus", "butterfly", "camel", "can opener", "car", "cart", 
					"cattle", "cello", "centipede", "chain saw", "chair", "chime", "cocktail shaker", "coffee maker", "computer keyboard", 
					"computer mouse", "corkscrew", "cream", "croquet ball", "crutch", "cucumber", "cup or mug", "diaper", "digital clock", "dishwasher", 
					"dog", "domestic cat", "dragonfly", "drum", "dumbbell", "electric fan", "elephant", "face powder", "fig", "filing cabinet", 
					"flower pot", "flute", "fox", "french horn", "frog", "frying pan", "giant panda", "goldfish", "golf ball", "golfcart", 
					"guacamole", "guitar", "hair dryer", "hair spray", "hamburger", "hammer", "hamster", "harmonica", "harp", "hat with a wide brim", 
					"head cabbage", "helmet", "hippopotamus", "horizontal bar", "horse", "hotdog", "iPod", "isopod", "jellyfish", "koala bear", "ladle", 
					"ladybug", "lamp", "laptop", "lemon", "lion", "lipstick", "lizard", "lobster", "maillot", "maraca", "microphone", "microwave", "milk can", 
					"miniskirt", "monkey", "motorcycle", "mushroom", "nail", "neck brace", "oboe", "orange", "otter", "pencil box", 
					"pencil sharpener", "perfume", "person", "piano", "pineapple", "ping-pong ball", "pitcher", "pizza", "plastic bag", 
					"plate rack", "pomegranate", "popsicle", "porcupine", "power drill", "pretzel", "printer", "puck", "punching bag", "purse", 
					"rabbit", "racket", "ray", "red panda", "refrigerator", "remote control", "rubber eraser", "rugby ball", "ruler", "salt or pepper shaker", 
					"saxophone", "scorpion", "screwdriver", "seal", "sheep", "ski", "skunk", "snail", "snake", "snowmobile", 
					"snowplow", "soap dispenser", "soccer ball", "sofa", "spatula", "squirrel", "starfish", "stethoscope", "stove", "strainer", 
					"strawberry", "stretcher", "sunglasses", "swimming trunks", "swine", "syringe", "table", "tape player", "tennis ball", "tick", 
					"tie", "tiger", "toaster", "traffic light", "train", "trombone", "trumpet", "turtle", "tv or monitor", "unicycle", 
					"vacuum", "violin", "volleyball", "waffle iron", "washer", "water bottle", "watercraft", "whale", "wine bottle", "zebra"]

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
			label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
			#label = '{}: {:.2f}'.format(int(box[0]), box[1])
			current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
			current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
			print("\t{} {}".format(classes[int(box[0])], box[1]))
			fp.write("\t{} {}\n".format(classes[int(box[0])], box[1]))
		print("\n")
		fp.write("\n")

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