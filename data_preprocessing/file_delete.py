
import numpy as np
import SimpleITK as sitk
import os
import glob
from skimage.transform import resize
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import time
from rembg import remove
from PIL import Image
import SimpleITK as sitk
import os

train_path = r'C:\Users\2418015\Downloads\HecktorData\training\imagesTr'
test_path = r'C:\Users\2418015\Downloads\HecktorData\testing\imagesTs'


ct_files = sorted(glob.glob(os.path.join(train_path, "CHUM-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(train_path, "CHUM-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\training\CHUM.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(train_path, "CHUP-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(train_path, "CHUP-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\training\CHUP.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(train_path, "CHUV-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(train_path, "CHUV-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\training\CHUV.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(train_path, "HGJ-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(train_path, "HGJ-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\training\HGJ.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(train_path, "HMR-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(train_path, "HMR-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\training\HMR.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(train_path, "MDA-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(train_path, "MDA-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\training\MDA(train).txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(test_path, "CHB-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(test_path, "CHB-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\testing\CHB.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(test_path, "MDA-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(test_path, "MDA-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\testing\MDA(testing).txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)

ct_files = sorted(glob.glob(os.path.join(test_path, "USZ-*__CT.nii.gz")))
pt_files = sorted(glob.glob(os.path.join(test_path, "USZ-*__PT.nii.gz")))
numb_directory = r'C:\Users\2418015\Downloads\HecktorData\testing\USZ.txt'
objects = open(numb_directory, 'r').read().splitlines()
nums = objects[0].split(',')

for i in range(len(nums)):
    rem_number = nums[i]
    rem = ct_files[int(rem_number)]
    os.remove(rem)
    rem = pt_files[int(rem_number)]
    os.remove(rem)
