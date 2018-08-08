import os
import shutil
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model_weights = "imagenet"
model = ResNet50(weights=model_weights)

target_size = (224, 224)

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
PATH_TO_OUTPUT_DIR = os.path.join(CWD_PATH, "output_images", model_weights)
if os.path.isdir(PATH_TO_OUTPUT_DIR):
	shutil.rmtree(PATH_TO_OUTPUT_DIR)
os.mkdir(PATH_TO_OUTPUT_DIR)

IMAGE_PATHS = []
for (dirpath, dirnames, filenames) in os.walk(PATH_TO_IMAGES_DIR):
	IMAGE_PATHS.extend(filenames)
	break

for image_path in IMAGE_PATHS:
	img = Image.open(os.path.join(PATH_TO_IMAGES_DIR, image_path))
	predictions = predict(model, img, target_size)
	print image_path.split("/")[-1], predictions
	# TODO: implement better output format such as bounding boxes etc.