import random
import numpy as np
import SimpleITK as sitk
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Lits_DataSet(Dataset):
    def __init__(self,root,stage='train'):
        self.root = root
        self.stage = stage
        f = open(os.path.join(self.root,self.stage+'.txt'))
        self.filename = f.read().splitlines()


    def __getitem__(self, index):
        objects = self.filename[index].replace("'","")
        sample, slice = objects.split('_')[0], np.uint16(objects.split('_')[1])
        PET = self.load(os.path.join(self.root,'datasets',sample,'PET.nii.gz'))
        CT = self.load(os.path.join(self.root,'datasets', sample, 'CT.nii.gz'))
        pet, ct = PET[slice, :, :], CT[slice, :, :]

        return pet[np.newaxis,:],ct[np.newaxis,:]

    def __len__(self):
        return len(self.filename)
        
    def load(self,file):
        itkimage = sitk.ReadImage(file)
        image = sitk.GetArrayFromImage(itkimage)
        return image    
    

if __name__ =='__main__':
    path = r'C:\Users\2418015\Downloads\Data\128'
    train_dataset = Lits_DataSet(path,'train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=25, num_workers=1, shuffle=True)
    for i, (PET,CT) in enumerate(train_dataloader):
        print(i,PET.shape,CT.shape)