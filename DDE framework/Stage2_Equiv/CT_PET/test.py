import torch
import numpy as np
import SimpleITK as sitk
from options.Options import Options_x
from dataset.dataset_lits_train import Lits_DataSet
from dataset.dataset_lits_test import Test_Datasets
from Model.model import ResUNet
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

    model = ResUNet(1, 1).to(device)

    ckpt = torch.load(opt.checkpoints_dir + '/' + opt.task_name + '/model/' + parameter_path) # find and load the saved parameters into the model
    model.load_state_dict(ckpt['model'])

    #log_test = logger.Test_Logger(os.path.join(opt.checkpoints_dir,opt.task_name,'logger'), "test_log")
    #save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name, 'result')
    log_test = logger.Test_Logger(r'C:\Users\2418015\Downloads\Project_Results\Stage2_E\CT-PET_3\logger',"test_CT-PET_log")
    save_result_path = (r'C:\Users\2418015\Downloads\Project_Results\Stage2_E\CT-PET_3')
    util.mkdir(save_result_path)

    datasets = Test_Datasets(opt.testing_datapath)
    for img_dataset, file_idx in datasets:

        PET, CT= img_dataset[0].to(device), img_dataset[1].to(device)
        PET = PET.unsqueeze(0).type(torch.float32)

        fake_PET = model(CT)

        fake_PET = fake_PET.cpu().detach().numpy().squeeze() *255
        real_PET = PET.cpu().detach().numpy().squeeze() *255

        NRMSE = normalized_root_mse(real_PET,fake_PET)
        PSNR = get_psnr(real_PET,fake_PET,255)
        SSIM = structural_similarity(real_PET,fake_PET, data_range=(255))

        log_test.update(file_idx,OrderedDict({'PSNR': PSNR,'SSIM': SSIM,'NRMSE': NRMSE}))
        cv2.imwrite(os.path.join(save_result_path,file_idx+'_fake.png'),fake_PET)
        cv2.imwrite(os.path.join(save_result_path,file_idx+'_real.png'),real_PET)

if __name__ == '__main__':
    test_result('best_model_img.pth') # C:\Users\2418015\OneDrive - University of Dundee\Documents\4th Year\Project\Code\pythonProject\Dual-domain Framework\Stage1\CTimg2PETimg\checkpoints\CT_img-PET_img\model
