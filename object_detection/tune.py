import os
import sys
import h5py
import numpy as np
import shutil

CWD_PATH = "/".join(os.path.realpath(__file__).split('/')[:-1])
DATA_PATH = os.path.join(CWD_PATH, "data", "ssd_keras")
sys.path.insert(0, DATA_PATH)
from misc_utils.tensor_sampling_utils import sample_tensors

weights_path = os.path.join(CWD_PATH, "weights", "VGG_ilsvrc15_SSD_500x500_iter_480000.h5")
tuned_weights_path = os.path.join(CWD_PATH, "weights", "tuned_VGG_ilsvrc15_SSD_500x500_iter_480000.h5")

shutil.copy(weights_path, tuned_weights_path)

weights_file = h5py.File(weights_path, 'r')
tuned_weights_file = h5py.File(tuned_weights_path)

def tune(classifier_name, init_shape, final_shape):
	kernel = weights_file[classifier_name][classifier_name]['kernel:0'].value
	bias = weights_file[classifier_name][classifier_name]['bias:0'].value

	if init_shape == kernel.shape:
		new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
											sampling_instructions=[final_shape[0], final_shape[1], final_shape[2], final_shape[3]],
											axes=[[3]], # The one bias dimension corresponds to the last kernel dimension.
											init=['gaussian', 'zeros'],
											mean=0.0,
											stddev=0.005)

		del tuned_weights_file[classifier_name][classifier_name]['kernel:0']
		del tuned_weights_file[classifier_name][classifier_name]['bias:0']
		tuned_weights_file[classifier_name][classifier_name].create_dataset(name='kernel:0', data=new_kernel)
		tuned_weights_file[classifier_name][classifier_name].create_dataset(name='bias:0', data=new_bias)

		print("{}, shape is tuned:\n\tKernel: {} -> {}\n\tBias: {} -> {}\n".format(classifier_name, kernel.shape, new_kernel.shape, bias.shape, new_bias.shape))


tune("conv4_3_norm_mbox_loc", (3,3,512,12), (3,3,512,16))

tune("conv4_3_norm_mbox_conf", (3, 3, 512, 603), (3, 3, 512, 804))

tune("conv9_2_mbox_loc", (3, 3, 256, 24), (3, 3, 256, 16))

tune("conv9_2_mbox_conf", (3, 3, 256, 1206), (3, 3, 256, 804))