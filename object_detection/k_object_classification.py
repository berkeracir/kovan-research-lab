import os
import shutil
import numpy as np
from PIL import Image
from natsort import natsorted
from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model_name = "ResNet50_imagenet"
model = ResNet50(weights="imagenet")
target_size = (224, 224)

#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
#model_name = "InceptionResNetV2_imagenet"
#model = InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#target_size = (299, 299)

def predict(model, img, target_size, top_n=3):
	"""Run model prediction on image
	Args:
		model: keras model
		img: PIL format image
		target_size: (width, height) tuple
		top_n: # of top predictions to return
	Returns:
		list of predicted labels and their probabilities
	"""

	if img.size != target_size:
		img = img.resize(target_size)

	preprocessed_img = image.img_to_array(img)
	preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
	preprocessed_img = preprocess_input(preprocessed_img)

	predictions = model.predict(preprocessed_img)
	return decode_predictions(predictions, top=top_n)[0]


CWD_PATH = "/".join(os.path.realpath(__file__).split('/')[:-1])

PATH_TO_IMAGES_DIR = os.path.join(CWD_PATH, "input_images")
PATH_TO_OUTPUT_DIR = os.path.join(CWD_PATH, "output")
#if os.path.isdir(PATH_TO_OUTPUT_DIR):
#	shutil.rmtree(PATH_TO_OUTPUT_DIR)
#os.mkdir(PATH_TO_OUTPUT_DIR)

IMAGE_PATHS = []
for (dirpath, dirnames, filenames) in os.walk(PATH_TO_IMAGES_DIR):
	IMAGE_PATHS.extend(natsorted(filenames))
	break

with open(os.path.join(PATH_TO_OUTPUT_DIR, model_name + ".txt"), 'w') as fp:
	for image_path in IMAGE_PATHS:
		img = Image.open(os.path.join(PATH_TO_IMAGES_DIR, image_path))
		predictions = predict(model, img, target_size)
		print image_path.split("/")[-1] + ":"
		fp.write(image_path.split("/")[-1] + ":\n")
		for pred in predictions:
			print "\t", pred
			fp.write('\t{} {} {}\n'.format(pred[0], pred[1], pred[2]))
		fp.write("\n")