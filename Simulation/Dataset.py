import torch.utils.data as tud
import random
import torch
import numpy as np
import scipy.io as sio


class dataset(tud.Dataset):
    def __init__(self, opt, HSI):
        super(dataset, self).__init__()
        self.isTrain = opt.isTrain
        self.size = opt.size
        self.path = opt.data_path
        if self.isTrain == True:
            self.num = opt.trainset_num
        else:
            self.num = opt.testset_num
        self.HSI = HSI

        ## load mask
        data = sio.loadmat(opt.mask_path)
        self.mask = data['mask']
        self.mask_3d = np.tile(self.mask[:, :, np.newaxis], (1, 1, 28))

    def __getitem__(self, index):
        if self.isTrain == True:
            index1   = random.randint(0, 29)
            hsi  =  self.HSI[:,:,:,index1]
        else:
            index1 = index
            hsi = self.HSI[:, :, :, index1]

        ## image patch
        shape = np.shape(hsi)
        px = random.randint(0, shape[0] - self.size)
        py = random.randint(0, shape[1] - self.size)
        label = hsi[px:px + self.size:1, py:py + self.size:1, :]

        ## mask patch
        pxm = random.randint(0, 256 - self.size)
        pym = random.randint(0, 256 - self.size)
        mask_3d = self.mask_3d[pxm:pxm + self.size:1, pym:pym + self.size:1, :]

        if self.isTrain == True:

            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                label  =  np.rot90(label)

            # Random vertical Flip
            for j in range(vFlip):
                label = label[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                label = label[::-1, :, :].copy()

        temp = mask_3d * label
        temp_shift = np.zeros((self.size, self.size + (28 - 1) * 2, 28))
        temp_shift[:, 0:self.size, :] = temp
        for t in range(28):
            temp_shift[:, :, t] = np.roll(temp_shift[:, :, t], 2 * t, axis=1)
        meas = np.sum(temp_shift, axis=2)
        input = meas / 28 * 2

        label = torch.FloatTensor(label.copy()).permute(2,0,1)
        input = torch.FloatTensor(input.copy()).unsqueeze(2).permute(2,0,1)
        return input, label

    def __len__(self):
        return self.num
