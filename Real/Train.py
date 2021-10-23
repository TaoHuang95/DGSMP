from Model import HSI_CS
from Dataset import dataset
import torch.utils.data as tud
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__=="__main__":

    print("===> New Model")
    model = HSI_CS(Ch=28, stages=4)

    print("===> Setting GPU")
    model = dataparallel(model, 1)  # set the number of parallel GPUs

    ## Initialize weight
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.constant_(layer.bias, 0.0)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)
            # nn.init.constant_(layer.bias, 0.0)

    ## Model Config
    parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
    parser.add_argument('--data_path_CAVE', default='./Data/CAVE_512_28/', type=str,
                        help='path of data')
    parser.add_argument('--data_path_KAIST', default='./Data/KAIST/28/', type=str,
                        help='path of data')
    parser.add_argument('--mask_path', default='./Data/mask.mat', type=str,
                        help='path of mask')
    parser.add_argument("--size", default=96, type=int, help='the size of trainset image')
    parser.add_argument("--trainset_num", default=20000, type=int, help='total number of trainset')
    parser.add_argument("--testset_num", default=10, type=int, help='total number of testset')
    parser.add_argument("--seed", default=1, type=int, help='Random_seed')
    parser.add_argument("--batch_size", default=8, type=int, help='batch_size')
    parser.add_argument("--isTrain", default=True, type=bool, help='train or test')
    opt = parser.parse_args()

    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    print(opt)

    ## Load training data
    key = 'train_CAVE.txt'
    file_path1 = opt.data_path_CAVE + key
    file_list1 = loadpath(file_path1)
    CAVE = prepare_data_cave(opt.data_path_CAVE, file_list1, 30)

    key = 'train_KAIST.txt'
    file_path2 = opt.data_path_KAIST + key
    file_list2 = loadpath(file_path2)
    KAIST = prepare_data_KASIT(opt.data_path_KAIST, file_list2, 30)

    ## Load trained model
    initial_epoch = findLastCheckpoint(save_dir="./Checkpoint")  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join("./Checkpoint", 'model_%03d.pth' % initial_epoch))

    ## Loss function
    criterion = nn.L1Loss()

    ## optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.1)  # learning rates

    ## pipline of training
    for epoch in range(initial_epoch, 500):
        model.train()
        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=8, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        start_time = time.time()
        for i, (input, label) in enumerate(loader_train):
            input, label = Variable(input), Variable(label)
            input, label = input.cuda(), label.cuda()
            out = model(input)

            loss = criterion(out, label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % (1000) == 0:
                print('%4d %4d / %4d loss = %.10f time = %s' % (
                    epoch + 1, i, len(Dataset)// opt.batch_size, epoch_loss / ((i+1) * opt.batch_size), datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(Dataset), elapsed_time))
        torch.save(model, os.path.join("./Checkpoint", 'model_%03d.pth' % (epoch + 1)))
