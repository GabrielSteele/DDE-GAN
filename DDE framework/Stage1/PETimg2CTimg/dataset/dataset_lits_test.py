import numpy as np
import SimpleITK as sitk
import os
import torch

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
        pet, ct = np.float32(PET[slice,:,:]), np.float32(CT[slice,:,:])

        pet = pet[np.newaxis,:]
        ct = ct[np.newaxis,:]

        img_dataset = [torch.from_numpy(pet),torch.from_numpy(ct)]
        yield img_dataset, file

if __name__ == '__main__':
    path = r'C:\Users\2418015\Downloads\final_data\128_testing'
    gen = Test_Datasets(path,'test')
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    for img_dataset, file_idx in gen:
        PET, CT = img_dataset[0].to(device), img_dataset[1].to(device)
        print('Test PET max val:{} Test CT max val:{}'.format(PET.max(), CT.max()))
