from shutil import copy
from os import listdir, makedirs, rmdir
from os.path import join, exists
from numpy.random import choice
from shutil import rmtree

dataset_path = "inputs/dataset_extra" 

output_test = "inputs/test"
output_train = "inputs/train"
output_val = "inputs/val"

image_folder_name = "images"
label_folder_name = "labels"

images = listdir(join(dataset_path, image_folder_name))
labels = listdir(join(dataset_path, label_folder_name))
nb_images = len(images)

nb_images_train = int(nb_images*0.5)
nb_images_test = int(nb_images*0.25)
nb_images_val = nb_images - nb_images_train - nb_images_test

id_range = list(range(0, nb_images))

ids_images_train = choice(id_range, nb_images_train, replace = False).tolist()

for id in ids_images_train:
    id_range.remove(id)

ids_images_val = choice(id_range, nb_images_val, replace = False).tolist()

for id in ids_images_val:
    id_range.remove(id)

ids_images_test = choice(id_range, nb_images_test, replace = False).tolist()


# Remove the folders if they exist.
if exists(join(output_train, image_folder_name)):
    rmtree(join(output_train, image_folder_name))
    rmtree(join(output_train, label_folder_name))

if exists(join(output_val, image_folder_name)):
    rmtree(join(output_val, image_folder_name))
    rmtree(join(output_val, label_folder_name))

if exists(join(output_test, image_folder_name)):
    rmtree(join(output_test, image_folder_name))
    rmtree(join(output_test, label_folder_name))


# Create the folders if they don't exist.
makedirs(join(output_train, image_folder_name))
makedirs(join(output_train, label_folder_name))

makedirs(join(output_val, image_folder_name))
makedirs(join(output_val, label_folder_name))

makedirs(join(output_test, image_folder_name))
makedirs(join(output_test, label_folder_name))


print("Nb Training images:", nb_images_train, "Nb Test Images:", nb_images_test, "Nb Validation Images:", nb_images_val)

# Copy the images and labels to the new folders.
for id in ids_images_train:
    copy(join(dataset_path, image_folder_name, images[id]), join(output_train, image_folder_name, images[id]))
    copy(join(dataset_path, label_folder_name, labels[id]), join(output_train, label_folder_name, labels[id]))

for id in ids_images_val:
    copy(join(dataset_path, image_folder_name, images[id]), join(output_val, image_folder_name, images[id]))
    copy(join(dataset_path, label_folder_name, labels[id]), join(output_val, label_folder_name, labels[id]))

for id in ids_images_test:
    copy(join(dataset_path, image_folder_name, images[id]), join(output_test, image_folder_name, images[id]))
    copy(join(dataset_path, label_folder_name, labels[id]), join(output_test, label_folder_name, labels[id]))
