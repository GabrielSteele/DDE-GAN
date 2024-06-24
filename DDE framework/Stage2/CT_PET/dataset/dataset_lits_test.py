import numpy as np
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt

def load(file):
    itkimage = sitk.ReadImage(file)
    image = sitk.GetArrayFromImage(itkimage)
    return image
    
def Test_Datasets(dataset_path,stage = 'test'):
    f = open(os.path.join(dataset_path, stage + '.txt'))
    data_list = f.read().splitlines()

    print("The number of test samples is: ", len(data_list))
    for file in data_list:
        print("\nStart Evaluate: ", file)
        objects = file.replace("'", "")
        sample, slice = objects.split('_')[0], np.uint16(objects.split('_')[1])

        PET = load(os.path.join(dataset_path, 'datasets', sample, 'PET.nii.gz'))
        CT = load(os.path.join(dataset_path, 'datasets', sample, 'CT.nii.gz'))
        pet, ct = (PET[slice, :, :]), (CT[slice, :, :])

        pet = pet[np.newaxis,np.newaxis,:]
        ct = ct[np.newaxis,np.newaxis,:]

        img_dataset = [torch.from_numpy(pet),torch.from_numpy(ct)]
        yield img_dataset, file