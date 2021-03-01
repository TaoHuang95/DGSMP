import torch
import torch.utils.data as tud
import os
import argparse
from Utils import *
import scipy.io as sio
import numpy as np
from Dataset import dataset
from torch.autograd import Variable
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='E:/实验区/TSA-Net-master/TSA-Net-master/TSA_Net_simulation/Data/Testing_data/', type=str,help='path of data')
parser.add_argument('--mask_path', default='G:/实验结果/2020-08_光谱压缩重构/Data/Train/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=256, type=int, help='the size of trainset image')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=10, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
opt = parser.parse_args()
print(opt)

def prepare_data_test(path, file_num):
    HR_HSI = np.zeros((((256,256,28,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path)  + 'scene%02d.mat' % (idx+1)
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['img']
    HR_HSI[HR_HSI < 0.] = 0.
    HR_HSI[HR_HSI > 1.] = 1.
    return HR_HSI


key = 'test_set.txt'

HR_HSI = prepare_data_test(opt.data_path,  10)

dataset = dataset(opt, HR_HSI)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)


for i in range(194, 195):  # number of model.pth
    model = torch.load("./checkpoint/model_%03d.pth"%i)
    model = model.eval()
    model = dataparallel(model, 1)
    psnr_total = 0
    k = 0
    for j, (input, label) in enumerate(loader_train):
        with torch.no_grad():
            input, label = Variable(input), Variable(label)
            input, label = input.cuda(), label.cuda(),

            start = time.time()
            out = model(input)
            elapsed = (time.time() - start)
            # print('Running time: {} Seconds'.format(elapsed))

            result = out
            result = result.clamp(min=0.,max=1.)

        psnr = compare_psnr(result.cpu().numpy(), label.cpu().numpy(), data_range=1.0)
        psnr_total = psnr_total + psnr
        k = k + 1
        print(psnr)  # every test picture's psnr
        res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
        save_path = './result/' + str(j + 1) + '.mat'
        sio.savemat(save_path, {'res':res})

    print(k)
    print("model %d, Avg PSNR = %.4f" % (i, psnr_total/k))
