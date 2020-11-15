import os
from PIL import Image
import numpy

k = 1

i = 49
j = 1
os.mkdir('train')
os.mkdir('train/images/')
os.mkdir('train/masks/')
image = Image.new('L', (672, 672))
new_size = (672, 672)

while i != 131:
    if i < 100:
        while os.path.isfile('Patients_CT/0' + str(i) + '/brain/' + str(j) + '.jpg'):
            path = 'Patients_CT/0' + \
                str(i) + '/brain/' + str(j) + '_HGE_Seg.jpg'
            if os.path.isfile(path):
                old_im = Image.open(
                    'Patients_CT/0' + str(i) + '/brain/' + str(j) + '.jpg')
                new_im = Image.new("L", new_size)
                new_im.paste(old_im)
                new_im.save('train/images/' + str(k) + '.jpg', "JPEG")
                old_im = Image.open(
                    'Patients_CT/0' + str(i) + '/brain/' + str(j) + '_HGE_Seg.jpg')
                old_size = old_im.size
                new_im = Image.new("L", new_size)
                new_im.paste(old_im)
                new_im.save('train/masks/' + str(k) + '.jpg', "JPEG")
            else:
                old_im = Image.open(
                    'Patients_CT/0' + str(i) + '/brain/' + str(j) + '.jpg')
                old_size = old_im.size
                new_im = Image.new("L", new_size)
                new_im.paste(old_im)
                new_im.save('train/images/' + str(k) + '.jpg', "JPEG")
                image.save('train/masks/' + str(k) + '.jpg', "JPEG")
            j += 1
            k += 1
            print(k)
    if i >= 100:
        while os.path.isfile('Patients_CT/' + str(i) + '/brain/' + str(j) + '.jpg'):
            path = 'Patients_CT/' + \
                str(i) + '/brain/' + str(j) + '_HGE_Seg.jpg'
            if os.path.isfile(path):
                old_im = Image.open(
                    'Patients_CT/' + str(i) + '/brain/' + str(j) + '.jpg')
                old_size = old_im.size
                new_im = Image.new("L", new_size)
                new_im.paste(old_im)
                new_im.save('train/images/' + str(k) + '.jpg', "JPEG")
                old_im = Image.open(
                    'Patients_CT/' + str(i) + '/brain/' + str(j) + '_HGE_Seg.jpg')
                old_size = old_im.size
                new_im = Image.new("L", new_size)
                new_im.paste(old_im)
                new_im.save('train/masks/' + str(k) + '.jpg', "JPEG")
            else:
                old_im = Image.open(
                    'Patients_CT/' + str(i) + '/brain/' + str(j) + '.jpg')
                old_size = old_im.size
                new_im = Image.new("L", new_size)
                new_im.paste(old_im)
                new_im.save('train/images/' + str(k) + '.jpg', "JPEG")
                image.save('train/masks/' + str(k) + '.jpg', "JPEG")
            j += 1
            k += 1
            print(k)
    j = 1
    i += 1
