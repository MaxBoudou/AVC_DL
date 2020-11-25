import os
import shutil
from numpy import random

def shuffle_dataset(prob, source_folder, source_images, source_masks, dest_folder):
	
	# masks = os.listdir(source_folder + source_masks)
	if source_folder[-1] != '/':
		source_folder + '/'
	if source_images[-1] != '/':
		source_images += '/'
	if source_masks[-1] != '/':
		source_masks += '/' 
	if dest_folder[-1] != '/':
		dest_folder += '/' 
	if not os.path.exists(dest_folder):
		os.makedirs(dest_folder)
		os.makedirs(dest_folder+'images/')
		os.makedirs(dest_folder+'masks/')
  


	files = os.listdir(source_folder + source_images)
	for f in files :
		if random.rand(1) < prob:
			shutil.move(source_folder+source_images+f, dest_folder+'images/')
			shutil.move(source_folder+source_masks+f, dest_folder+'masks/')

	return (dest_folder+'train/', dest_folder+'test/')