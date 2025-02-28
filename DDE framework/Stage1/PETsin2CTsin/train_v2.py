import torch.optim as optim
from utils.common import adjust_learning_rate
from utils.metrics import LossAverage
from test import *
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


def train(train_dataloader, epoch):
    print("=======Epoch:{}===================================".format(epoch))
    Loss = LossAverage()
    NRMSE = LossAverage()
    PSNR = LossAverage()
    SSIM = LossAverage()
    model.train()

    l1_loss = nn.L1Loss()

    for i, (PET, CT) in enumerate(train_dataloader):  # inner loop within one epoch
        PET = PET.to(device)
        CT = CT.to(device)

        PET = FP(PET).type(torch.FloatTensor).detach().to(device)
        CT = FP(CT).type(torch.FloatTensor).detach().to(device)

        fake_CT = model(PET)
        loss = l1_loss(CT, fake_CT)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        adjust_learning_rate(optimizer, epoch, opt, opt.step)
        Real_ct = CT.cpu().detach().numpy().squeeze() * 255
        Fake_ct = fake_CT.cpu().detach().numpy().squeeze() * 255

        nrmse = normalized_root_mse(Real_ct, Fake_ct)
        psnr = get_psnr(Real_ct, Fake_ct, 255)
        ssim = get_ssim(Real_ct, Fake_ct)

        Loss.update(loss.item(), 1)
        NRMSE.update(nrmse, 1)
        PSNR.update(psnr, 1)
        SSIM.update(ssim, 1)

    return OrderedDict(
        {'Train_Loss': Loss.avg, 'Train_PSNR': PSNR.avg, 'Train_SSIM': SSIM.avg, 'Train_NRMSE': NRMSE.avg})

def val(val_dataloader):
    Loss = LossAverage()
    NRMSE = LossAverage()
    PSNR = LossAverage()
    SSIM = LossAverage()
    model.eval()

    l1_loss = nn.L1Loss()


    for i, (PET, CT) in enumerate(val_dataloader):  # inner loop within one epoch
        PET = PET.to(device)
        CT = CT.to(device)

        PET = FP(PET).type(torch.FloatTensor).detach().to(device)
        CT = FP(CT).type(torch.FloatTensor).detach().to(device)
        with torch.no_grad():
            fake_CT = model(PET)
            loss = l1_loss(CT, fake_CT)

        Real_ct = CT.cpu().detach().numpy().squeeze() * 255
        Fake_ct = fake_CT.cpu().detach().numpy().squeeze() * 255

        nrmse = normalized_root_mse(Real_ct, Fake_ct)
        psnr = get_psnr(Real_ct, Fake_ct, 255)
        ssim = get_ssim(Real_ct, Fake_ct)

        Loss.update(loss.item(), 1)
        NRMSE.update(nrmse, 1)
        PSNR.update(psnr, 1)
        SSIM.update(ssim, 1)

    return OrderedDict({'Val_Loss': Loss.avg, 'Val_PSNR': PSNR.avg, 'Val_SSIM': SSIM.avg, 'Val_NRMSE': NRMSE.avg})


if __name__ == '__main__':
    opt = Options_x().parse()  # get training options
    device = torch.device('cuda:' + opt.gpu_ids if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model = ResNet(1, 1).to(device)
    FP = Forward_projection('astra_cuda').to(device)
    BP = Backward_projection('astra_cuda').to(device)

    train_dataset = Lits_DataSet(opt.datapath, 'train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, \
                                  num_workers=opt.num_threads, shuffle=True)
    val_dataset = Lits_DataSet(opt.datapath, 'val')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, \
                                num_workers=opt.num_threads, shuffle=True)

    save_result_path = os.path.join(opt.checkpoints_dir, opt.task_name)
    util.mkdir(save_result_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    best = [0, np.inf]

    model_save_path = os.path.join(save_result_path, 'model')
    util.mkdir(model_save_path)
    logger_save_path = os.path.join(save_result_path, 'logger')
    util.mkdir(logger_save_path)

    log_train = logger.Train_Logger(logger_save_path, "train_log")

    for epoch in range(opt.epoch):
        epoch = epoch + 1
        train_log = train(train_dataloader, epoch)
        val_log = val(val_dataloader)

        log_train.update(epoch, train_log, val_log)

        state = {'model': model.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(model_save_path, 'latest_model.pth'))
        if val_log['Val_Loss'] < best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(model_save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_Loss']

        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

        if epoch % opt.model_save_fre == 0:
            torch.save(state, os.path.join(model_save_path, 'model_' + str(epoch) + '.pth'))

        torch.cuda.empty_cache()

    test_result('best_model.pth')