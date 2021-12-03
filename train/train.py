# coding=utf-8
from data_read.data_read import DatasetFromFolder, TestFromFolder
import time
import argparse
from cal_ssim import SSIM
import os
import numpy as np
import torch
from torch import nn, optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.autograd import Variable
from model.Net import Generator
from utils_set.utils import save_checkpoint_val, adjust_learning_rate_40, print_log, save_checkpoint_val_best, \
    print_log_test

# Training settings
parser = argparse.ArgumentParser(description="PyTorch NHRN")
parser.add_argument("--tag", type=str, help="tag for this training", default='ITS')
parser.add_argument("--train", default="", type=str,
                    help="path to load train datasets(default: none)")
parser.add_argument("--test", default="", type=str,
                    help="path to load test datasets(default: none)")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=120, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="step to test the model performance. Default=2000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?", default='--cuda')
parser.add_argument("--gpus", type=int, default=1, help="nums ofz gpu to use")
parser.add_argument("--resume", default=" ", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--txt_name", type=str, help="the name of saving the txt", default='ITS')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    global opt, name, logger, model, criterion, SSIM_loss, start_time, txt_name
    opt = parser.parse_args()
    print(opt)

    name = "%s_%d" % (opt.tag, opt.batchSize)

    txt_name = "%s" % (opt.txt_name)

    if not os.path.exists("training_log"):
        os.makedirs("training_log")

    if not os.path.exists("runs"):
        os.makedirs("runs")

    logger = SummaryWriter("runs/" + name)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = 1334
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = False

    print("==========> Loading datasets")

    train_dataset = DatasetFromFolder(opt.train, transform=Compose([
        ToTensor()
    ]))

    indoor_test_dataset = TestFromFolder(opt.test, transform=Compose([
        ToTensor()
    ]))

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True)
    indoor_test_loader = DataLoader(dataset=indoor_test_dataset, num_workers=opt.threads, batch_size=1, pin_memory=True,
                                    shuffle=True)

    print("==========> Building model")

    model = Generator()
    criterion = nn.MSELoss(size_average=True)
    SSIM_loss = SSIM()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['state_dict'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("==========> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        SSIM_loss = SSIM_loss.cuda()

    else:
        model = model.cpu()
        criterion = criterion.cpu()
        SSIM_loss = SSIM_loss.cpu()

    print("==========> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    psnr_old, ssim_old = test(indoor_test_loader, 0)

    print("==========> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):

        start_time = time.time()
        train(training_data_loader, optimizer, epoch)
        save_checkpoint_val(model, epoch, name, txt_name)
        val_psnr, val_ssim = test(indoor_test_loader, epoch)
        if val_psnr >= psnr_old:
            save_checkpoint_val_best(model, epoch, name, txt_name)
            psnr_old = val_psnr


def train(training_data_loader, optimizer, epoch):
    adjust_learning_rate_40(optimizer, epoch)
    print("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    for iteration, batch in enumerate(training_data_loader, 1):

        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        steps = len(training_data_loader) * (epoch - 1) + iteration

        data, label = Variable(batch[0]), \
                      Variable(batch[1])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()

        else:
            data = data.cpu()
            label = label.cpu()

        output = model(data)

        output_tab = torch.clamp(output, 0, 1)

        loss1 = criterion(output, label)
        loss2 = 1 - SSIM_loss(output, label)

        loss = loss1 + 0.1 * loss2
        loss.backward()

        optimizer.step()

        if iteration % 10 == 0:
            one_time = time.time() - start_time
            print_log(one_time, epoch, iteration, len(training_data_loader),
                      loss.item(), loss1.item(), 1 - loss2.item(), txt_name)

            logger.add_scalar('loss', loss.item(), steps)
            logger.add_scalar('image_loss', loss1.item(), steps)

        if iteration % opt.step == 0:
            data_temp = make_grid(data.data)
            label_temp = make_grid(label.data)
            output_temp = make_grid(output_tab.data)

            logger.add_scalar('train_psnr_output', 10 * np.log10(1.0 / loss1.item()), steps)
            logger.add_image('temp_data', data_temp, steps)
            logger.add_image('temp_output_label', label_temp, steps)
            logger.add_image('temp_output', output_temp, steps)

        torch.cuda.empty_cache()


def test(test_data_loader, epoch):
    psnrs = []
    mses = []
    ssims = []
    for iteration, batch in enumerate(test_data_loader, 1):
        model.eval()
        data, label = \
            Variable(batch[0]), \
            Variable(batch[1])

        if opt.cuda:
            data = data.cuda()
            label = label.cuda()
        else:
            data = data.cpu()
            label = label.cpu()

        with torch.no_grad():
            output = model(data)

        output = torch.clamp(output, 0., 1.)

        mse = nn.MSELoss()(output, label)
        ssim = SSIM()(output, label)
        mses.append(mse.item())
        ssims.append(ssim.item())
        psnr = 10 * np.log10(1.0 / mse.item())
        psnrs.append(psnr)
    psnr_mean = np.mean(psnrs)
    mse_mean = np.mean(mses)
    ssim_mean = np.mean(ssims)

    print_log_test(epoch, psnr_mean, ssim_mean, txt_name)

    logger.add_scalar('psnr', psnr_mean, epoch)
    logger.add_scalar('mse', mse_mean, epoch)
    logger.add_scalar('ssim', ssim_mean, epoch)

    data = make_grid(data.data)
    label = make_grid(label.data)
    output = make_grid(output.data)

    logger.add_image('val_data', data, epoch)
    logger.add_image('val_label', label, epoch)
    logger.add_image('val_output', output, epoch)
    return psnr_mean, ssim_mean


if __name__ == "__main__":
    os.system('clear')
    main()
