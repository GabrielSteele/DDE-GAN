import numpy as np
import SimpleITK as sitk
import os
import glob
from skimage.transform import resize
import nibabel as nib
from rembg import remove
from PIL import Image

output_root = r'C:\Users\2418015\Downloads\Final_Data\128_training'
path = r'C:\Users\2418015\Downloads\Selected_Data\training\imagesTr'
all_files = glob.glob(os.path.join(path, '*.nii.gz'))
stage = 'train'

chum_files = [file for file in all_files if "CHUM" in os.path.basename(file)]
chup_files = [file for file in all_files if "CHUP" in os.path.basename(file)]
chuv_files = [file for file in all_files if "CHUV" in os.path.basename(file)]
hgj_files = [file for file in all_files if "HGJ" in os.path.basename(file)]
hmr_files = [file for file in all_files if "HMR" in os.path.basename(file)]
mda_files = [file for file in all_files if "MDA" in os.path.basename(file)]

ct_chum_files = [file for file in chum_files if file.endswith("CT.nii.gz")]
pet_chum_files = [file for file in chum_files if file.endswith("PT.nii.gz")]
ct_chup_files = [file for file in chup_files if file.endswith("CT.nii.gz")]
pet_chup_files = [file for file in chup_files if file.endswith("PT.nii.gz")]
ct_chuv_files = [file for file in chuv_files if file.endswith("CT.nii.gz")]
pet_chuv_files = [file for file in chuv_files if file.endswith("PT.nii.gz")]
ct_hgj_files = [file for file in hgj_files if file.endswith("CT.nii.gz")]
pet_hgj_files = [file for file in hgj_files if file.endswith("PT.nii.gz")]
ct_hmr_files = [file for file in hmr_files if file.endswith("CT.nii.gz")]
pet_hmr_files = [file for file in hmr_files if file.endswith("PT.nii.gz")]
ct_mda_files = [file for file in mda_files if file.endswith("CT.nii.gz")]
pet_mda_files = [file for file in mda_files if file.endswith("PT.nii.gz")]


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

for a in range(len(ct_chum_files)):
    pet_data = nib.load(pet_chum_files[a]).get_fdata()
    pet_new = trim(pet_data, 5, 10)
    ct_data = nib.load(ct_chum_files[a]).get_fdata()
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


for b in range(len(ct_chup_files)):
    pet_data = nib.load(pet_chup_files[b]).get_fdata()
    pet_data = trim(pet_data, 25, 10)
    ct_data = nib.load(ct_chup_files[b]).get_fdata()
    ct_data = trim(ct_data, 25, 10)

    ct_images, pet_images = process(ct_data, pet_data, cut=360)

    output_dir = os.path.join(output_root, 'datasets', str(a+b))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b, pet_data.shape[-1]))

for c in range(len(ct_chuv_files)):
    pet_data = nib.load(pet_chuv_files[c]).get_fdata()
    pet_data = trim(pet_data, 15, 15)
    ct_data = nib.load(ct_chuv_files[c]).get_fdata()
    ct_data = trim(ct_data, 15, 15)

    ct_images, pet_images = process(ct_data, pet_data, cut=350)

    output_dir = os.path.join(output_root, 'datasets', str(a+b+c))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b+c, ct_data.shape[-1]))

for d in range(len(ct_hgj_files)):
    pet_data = nib.load(pet_hgj_files[d]).get_fdata()
    pet_data = trim(pet_data, 3, 15)
    ct_data = nib.load(ct_hgj_files[d]).get_fdata()
    ct_data = trim(ct_data, 3, 15)

    ct_images, pet_images = process(ct_data, pet_data, cut=310)

    output_dir = os.path.join(output_root, 'datasets', str(a+b+c+d))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b+c+d, ct_data.shape[-1]))

for e in range(len(ct_hmr_files)):
    pet_data = nib.load(pet_hmr_files[e]).get_fdata()
    pet_data = trim(pet_data, 8, 10)
    ct_data = nib.load(ct_hmr_files[e]).get_fdata()
    ct_data = trim(ct_data, 8, 10)

    ct_images, pet_images = process(ct_data, pet_data)

    output_dir = os.path.join(output_root, 'datasets', str(a+b+c+d+e))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b+c+d+e, ct_data.shape[-1]))

for f in range(len(ct_mda_files)):
    pet_data = nib.load(pet_mda_files[f]).get_fdata()
    pet_data = trim(pet_data, 15, 15)
    ct_data = nib.load(ct_mda_files[f]).get_fdata()
    ct_data = trim(ct_data, 15, 15)

    ct_images, pet_images = process(ct_data, pet_data, cut=370)

    output_dir = os.path.join(output_root, 'datasets', str(a+b+c+d+e+f))
    os.makedirs(output_dir, exist_ok=True)
    pet_output_path = os.path.join(output_dir, 'PET.nii.gz')
    ct_output_path = os.path.join(output_dir, 'CT.nii.gz')
    sitk.WriteImage(pet_images, pet_output_path)
    sitk.WriteImage(ct_images, ct_output_path)

    print("'{}_{}'".format(a+b+c+d+e+f, ct_data.shape[-1]))