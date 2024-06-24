import numpy as np
import SimpleITK as sitk
import os
import glob
from skimage.transform import resize
import nibabel as nib
from rembg import remove
from PIL import Image

output_root = r'C:\Users\2418015\Downloads\New_Data\testing_128'
path = r'C:\Users\2418015\Downloads\HecktorData\testing\imagesTs'
all_files = glob.glob(os.path.join(path, '*.nii.gz'))

chb_files = [file for file in all_files if "CHB" in os.path.basename(file)]
mda_files = [file for file in all_files if "MDA" in os.path.basename(file)]
usz_files = [file for file in all_files if "USZ" in os.path.basename(file)]

ct_chb_files = [file for file in chb_files if file.endswith("CT.nii.gz")]
pet_chb_files = [file for file in chb_files if file.endswith("PT.nii.gz")]
ct_mda_files = [file for file in mda_files if file.endswith("CT.nii.gz")]
pet_mda_files = [file for file in mda_files if file.endswith("PT.nii.gz")]
ct_usz_files = [file for file in usz_files if file.endswith("CT.nii.gz")]
pet_usz_files = [file for file in usz_files if file.endswith("PT.nii.gz")]

def trim(dataset, top, bottom):
    max = dataset.shape[-1]-top
    wind = max - bottom
    trimmed = dataset[:, :, wind:max]
    return trimmed

def process(ct_data, pet_data, cut = 0):
    ct_images = []
    pet_images = []
    for i in range(ct_data.shape[-1]):
        im = np.array(ct_data[:,:,i])
        if cut > 0:
            im[:,cut:] = 0
        im = Image.fromarray(im)
        im = remove(im)
        im = np.array(im)
        im = np.mean(im[:, :, :3], axis=2)
        im = resize(im, (128, 128), anti_aliasing=True)
        im = (im - np.min(im)) / (np.max(im) - np.min(im) + 0.000001)
        im = sitk.GetImageFromArray(im)
        ct_images.append(im)

        im = resize(pet_data[:, :, i], (128, 128), anti_aliasing=True)
        im = (im - np.min(im)) / (np.max(im) - np.min(im) + 0.000001)
        im = sitk.GetImageFromArray(im)
        pet_images.append(im)

    return sitk.JoinSeries(ct_images), sitk.JoinSeries(pet_images)

for a in range(len(ct_chb_files)):
    pet_data = nib.load(pet_chb_files[a]).get_fdata()
    pet_new = trim(pet_data, 5, 10)
    ct_data = nib.load(ct_chb_files[a]).get_fdata()
    ct_new = trim(ct_data, 5, 30)
    g = int((ct_new.shape[-1]) / 10)
    ct_max = ct_new.shape[-1]
    ct_new = ct_new[:, :, 0:ct_max:g]

    ct_images, pet_images = process(ct_new,pet_new)

    output_dir = os.path.join(output_root, 'datasets', str(a))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a,pet_new.shape[-1]))

for b in range(len(ct_mda_files)):
    pet_data = nib.load(pet_mda_files[b]).get_fdata()
    pet_data = trim(pet_data, 25, 10)
    ct_data = nib.load(ct_mda_files[b]).get_fdata()
    ct_data = trim(ct_data, 25, 10)

    ct_images, pet_images = process(ct_data, pet_data, cut=360)

    output_dir = os.path.join(output_root, 'datasets', str(a+b))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b, pet_data.shape[-1]))

for c in range(len(ct_usz_files)):
    pet_data = nib.load(pet_usz_files[c]).get_fdata()
    pet_data = trim(pet_data, 15, 15)
    ct_data = nib.load(ct_usz_files[c]).get_fdata()
    ct_data = trim(ct_data, 15, 15)

    ct_images, pet_images = process(ct_data, pet_data, cut=350)

    output_dir = os.path.join(output_root, 'datasets', str(a+b+c))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b+c, ct_data.shape[-1]))