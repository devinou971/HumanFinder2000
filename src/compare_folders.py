from os import listdir
from shutil import copy
from os.path import join
input_folder = "inputs/thermal0"
file_not_present_in = "inputs/label-studio-projet/label-studio-input"
output_folder = "inputs/label-studio-projet/extra-input"

input_files = listdir(input_folder)
sync_files = listdir(file_not_present_in)

for f in input_files:
    if  f not in sync_files:
        copy(join(input_folder, f), join(output_folder, f))
