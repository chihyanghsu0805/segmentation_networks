from __future__ import absolute_import
from __future__ import print_function

import csv

import nibabel as nib
import numpy as np

def write_history(history_dict, output_path):
  with open(output_path,'w') as f:
    writer = csv.writer(f)
    writer.writerow(history_dict.keys())
    writer.writerows(zip(*history_dict.values()))

def read_nifti_image(file_path):
  img_obj = nib.load(file_path)
  image = img_obj.get_fdata()

  return image

def read_nifti_header(file_path):
  img_obj = nib.load(file_path)
  header = img_obj.header

  return header

def write_nifti_image(file_path, image, header):
  if header:
    sform = header.get_sform()
    image_obj = nib.Nifti1Image(image, sform)
  else:
    image_obj = nib.Nifti1Image(image, None)
  nib.save(image_obj, file_path)

#def read_csv_label(csv_file):
#  with open(csv_file, 'r') as f:
#    reader = csv.reader(f)
#    label_list = list(reader)

#  label_list = [int(x[0]) for x in label_list]

#  return np.array(label_list)
