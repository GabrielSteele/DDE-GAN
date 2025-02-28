import torch
import numpy as np
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_test import Test_Datasets
from Model.Sin_model import ResNet
from Model.Projection_operator import Forward_projection,Backward_projection
from torch.utils.data import DataLoader
from utils import logger,util
import torch.nn as nn
from utils.common import get_ssim, get_psnr
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
import os
import cv2
from collections import OrderedDict

def test_result(parameter_path = 'latest_model.pth'):
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    model = ResNet(1, 1).to(device)
    FP = Forward_projection('astra_cuda').to(device)
    BP = Backward_projection('astra_cuda').to(device)

    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + parameter_path)
    model.load_state_dict(ckpt['model'])

    #log_test = logger.Test_Logger(os.path.join(opt.checkpoints_dir,opt.task_name,'logger'), "test_pet-ct_log")
    log_test = logger.Test_Logger(r'C:\Users\2418015\Downloads\Project_Results\Stage1_5\PETsin-CTsin\logger', "test_PETsin-CTsin_log")
    #save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'result')
    save_result_path = (r'C:\Users\2418015\Downloads\Project_Results\Stage1_5\PETsin-CTsin')
    util.mkdir(save_result_path)


    datasets = Test_Datasets(opt.testing_datapath)

    for img_dataset, file_idx in datasets:
        PET, CT= img_dataset[0].to(device), img_dataset[1].to(device)
        PET = PET.unsqueeze(0).type(torch.float32)
        CT = CT.unsqueeze(1).type(torch.float32)

        PET = FP(PET).type(torch.FloatTensor).detach().to(device)
        CT = FP(CT).type(torch.FloatTensor).detach().to(device)

        fake_CT = model(PET)

        fake_CT = fake_CT.cpu().detach().numpy().squeeze() *255
        real_CT = CT.cpu().detach().numpy().squeeze() *255
        PET_sin = PET.cpu().detach().numpy().squeeze() *255

        NRMSE = normalized_root_mse(real_CT,fake_CT)
        PSNR = get_psnr(real_CT,fake_CT,255)
        SSIM = structural_similarity(real_CT,fake_CT, data_range=255)

        log_test.update(file_idx,OrderedDict({'PSNR': PSNR,'SSIM': SSIM,'NRMSE': NRMSE}))
        cv2.imwrite(os.path.join(save_result_path, file_idx + '_input_PET.png'), PET_sin)
        cv2.imwrite(os.path.join(save_result_path,file_idx+'_fake.png'),fake_CT)
        cv2.imwrite(os.path.join(save_result_path,file_idx+'_real.png'),real_CT)

if __name__ == '__main__':
    test_result('best_model.pth')
