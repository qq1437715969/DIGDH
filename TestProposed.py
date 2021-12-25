# encoding: utf-8


import argparse
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms

from HidingNet import HidingNet
from RevealNet import RevealNet
import math

from ImageFolderDataset import MyKeyCoverAndSecretFolder
import gc
from PIL import Image
irange = range


def make_grid(tensor, nrow=8, padding=0,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def tensor2Img(tensor):
    from PIL import Image
    grid = make_grid(tensor, nrow=8, padding=2, pad_value=0,
                     normalize=True, range=None, scale_each=False)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # return Image.fromarray(ndarr)
    return ndarr

Hnet_path = 'models/netH_1_epoch_188,sumloss=0.00004583,Hloss=0.00000896.pth'
Rnet_path = 'models/netR_1_epoch_188,sumloss=0.00004583,Rloss=0.00004916.pth'

OLD_DATA_DIR = '/home/duanxt/Desktop/LY_Cycle_Gan/StegPro/image_net/'

test_key_imgs_path = 'test_key_imgs'

DATA_DIR = OLD_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=1,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=40,
                    help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=16,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=1000,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet_1', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet_1', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--summary_writer_path', default='./training/',
                    help='folder to save the summary_writer')
parser.add_argument('--betaR', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default=DATA_DIR, help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default='LY', help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=20, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=400, help='the frequency of save the resultPic')


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/main_gan_bak.py
    cur_work_dir, mainfile = os.path.split(main_file_path)  # eg：/n/liyz/videosteganography/

    print(cur_work_dir)
    file_names = os.listdir(cur_work_dir)
    for file_name in file_names:
        if os.path.splitext(file_name)[1] == '.py':  # 目录下包含.json的文件
            shutil.copyfile(cur_work_dir + "/" + file_name, des_path + "/" + file_name)


def main():
    gc.collect()
    torch.cuda.empty_cache()
    ############### define global parameters ###############
    global opt, optimizerH_1, optimizerH_2, optimizerR_1, optimizerR_2, optimizerR_1_2, writer, logPath, schedulerH_1, schedulerH_2, schedulerR_1, schedulerR_2, val_loader, smallestLoss, key_img_1_tensor, key_img_2_tensor

    #################  output configuration   ###############
    opt = parser.parse_args()

    transformer = transforms.Compose([
        transforms.Resize([opt.imageSize, opt.imageSize]),  # resize to a given size
        transforms.ToTensor(),
    ])

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create dirs to save the result #############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            opt.summary_writer_path += experiment_dir + "/summary_writer"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)
            if not os.path.exists(opt.summary_writer_path):
                os.makedirs(opt.summary_writer_path)

            if opt.test != '':
                imgs_dirs = ['cover', 'stego', 'secret_1', 'rev_secret_1', 'secret_2', 'rev_secret_2']
                test_imgs_dirs = [os.path.join(opt.testPics, x) for x in imgs_dirs]
                for path in test_imgs_dirs:
                    if not os.path.exists(path):
                        os.makedirs(path)
                opt.testPics = test_imgs_dirs

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)

    if test_key_imgs_path is None or test_key_imgs_path == '':
        print("key_img_path缺失")
        return
    opt.Hnet_1 = Hnet_path
    opt.Rnet_1 = Rnet_path

    test_dataset = MyKeyCoverAndSecretFolder(
        os.path.join(opt.test, test_key_imgs_path),
        # os.path.join(opt.test, test_key_imgs_path),
        # os.path.join(opt.test, 'test_key_imgs_2'),
        os.path.join(opt.test, "test_cov"),
        os.path.join(opt.test, "test_secret"),
        transformer
    )

    # Hnet_1 = FCDense(depths=4, growth_rates=12)
    Hnet_1 = HidingNet()
    Hnet_1.load_state_dict(torch.load(opt.Hnet_1))

    Rnet_1 = RevealNet()
    Rnet_1.load_state_dict(torch.load(opt.Rnet_1))

    if torch.cuda.is_available():

        Hnet_1 = Hnet_1.cuda()
        Rnet_1 = Rnet_1.cuda()

    # MSE loss
    criterion = nn.MSELoss().cuda()

    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=0, pin_memory=True)
    test(test_loader, 0, Hnet_1=Hnet_1, Rnet_1=Rnet_1, criterion=criterion)

    print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


# TODO 修改
def test(test_loader, epoch, Hnet_1, Rnet_1, criterion):
    print(
        "#################################################### test begin ########################################################")
    gc.collect()
    torch.cuda.empty_cache()
    start_time = time.time()
    Hnet_1.eval()
    Rnet_1.eval()

    Hlosses = AverageMeter()  # record the Hloss in one epoch
    Rlosses = AverageMeter()  # record the Rloss in one epoch
    this_batch_size = test_loader.batch_size
    for idx, (_, key_img, _, cover_img, _, secret_img) in enumerate(test_loader, 0):

        Hnet_1.zero_grad()
        Rnet_1.zero_grad()

        if opt.cuda:
            key_img = key_img.cuda()
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()

        key_img_with_secret = Hnet_1(torch.cat([key_img, secret_img], dim=1))
        err_img = key_img_with_secret - key_img
        # 3 stego
        stego_img = torch.clamp((cover_img + err_img), min=0, max=1)
        errH_0 = criterion(stego_img, cover_img)
        errH = errH_0
        Hlosses.update(errH.item(), this_batch_size)

        rev_sec = Rnet_1(stego_img - cover_img + key_img)
        errR_0 = criterion(rev_sec, secret_img)
        errR = errR_0

        Rlosses.update(errR.item(), this_batch_size)

        save_result_pic_test(this_batch_size, cover_img, stego_img.data, secret_img, rev_sec.data,
                             (Rnet_1(stego_img)).data, (Rnet_1(stego_img-cover_img)).data, 0, i=idx, save_path=opt.testPics)
        gc.collect()
        torch.cuda.empty_cache()

    test_hloss = Hlosses.avg
    test_rloss = Rlosses.avg
    test_sumloss = test_hloss + opt.beta * test_rloss

    test_time = time.time() - start_time
    test_log = "validation[%d] test_Hloss = %.8f\t test_Rloss = %.8f\t test_Sumloss = %.8f\t testidation time=%.2f" % (
        epoch, test_hloss, test_rloss, test_sumloss, test_time)
    print_log(test_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return test_hloss, test_rloss, test_sumloss


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print info onto the console
    if console:
        print(log_info)
    # debug mode will not write logs into files
    if not opt.debug:
        # write logs into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')

def save_result_pic_test(this_batch_size, originalLabelv, ContainerImg, secret_1_Labelv, RevSec_1_Img, secret_2_Labelv,
                         RevSec_2_Img, epoch, i, save_path):
    if not opt.debug:
        originalFrames = Image.fromarray(tensor2Img(originalLabelv), 'RGB')
        containerFrames = Image.fromarray(tensor2Img(ContainerImg), 'RGB')
        secret_1_Frames = Image.fromarray(tensor2Img(secret_1_Labelv), 'RGB')
        revSec_1_Frames = Image.fromarray(tensor2Img(RevSec_1_Img), 'RGB')
        secret_2_Frames = Image.fromarray(tensor2Img(secret_2_Labelv), 'RGB')
        revSec_2_Frames = Image.fromarray(tensor2Img(RevSec_2_Img), 'RGB')

        originalName = '%s/cover%d.png' % (save_path[0], i)
        originalFrames.save(originalName, quality=100, subsampling=0)
        containerName = '%s/stego%d.png' % (save_path[1], i)
        containerFrames.save(containerName, quality=100, subsampling=0)

        secret_1_Name = '%s/secret_1_%d.png' % (save_path[2], i)
        secret_1_Frames.save(secret_1_Name, quality=100, subsampling=0)
        revSec_1_Name = '%s/revSec_1_%d.png' % (save_path[3], i)
        revSec_1_Frames.save(revSec_1_Name, quality=100, subsampling=0)

        # secret_2_Name = '%s/secret_2_%d.png' % (save_path[4], i)
        # secret_2_Frames.save(secret_2_Name, quality=100, subsampling=0)
        # revSec_2_Name = '%s/revSec_2_%d.png' % (save_path[5], i)
        # revSec_2_Frames.save(revSec_2_Name, quality=100, subsampling=0)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    main()