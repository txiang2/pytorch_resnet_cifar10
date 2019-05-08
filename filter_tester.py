import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

import numpy as np

from PIL import Image

from torch.nn.functional import binary_cross_entropy_with_logits
import torchvision.transforms.functional

from filter_models import *

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--filter-weight', default=0.3, type=float, metavar='M',
                    help='wieght of filter (.4 weight = max 25 pixel intensity)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--filter-state', default='', type=str, metavar='PATH',
                    help='path to pre-trained filter')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
                    

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()
    
    filter_model = ConvFilter2(filter_weight=args.filter_weight)
    filter_model.cuda()
    
    filter_model.load_state_dict(torch.load(args.filter_state))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                        

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this implementation it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    prec1 = validate_filter(val_loader, model, filter_model, criterion, filter_model.filter_weight)


def un_normalize(tensor):

    tensor[:,0] = (tensor[:,0] * .229) + .485
    tensor[:,1] = (tensor[:,1] * .224) + .456
    tensor[:,2] = (tensor[:,2] * .225) + .406
    
    tensor = torch.max(tensor, torch.zeros(tensor.shape).cuda())
    tensor = torch.min(tensor, torch.ones(tensor.shape).cuda())
    
    return tensor


def validate_filter(val_loader, model, filter_model, criterion, noise_limit):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topn = AverageMeter()
    topp = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    confusion_matrix = np.zeros((10,10), dtype='int')
    
    ii = 0
    
    os.makedirs('tested_images', exist_ok=True)
    
    fooled = []
    
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)

        if args.half:
            input_var = input_var.half()
            
        
        adjusted_input = torch.FloatTensor(size=(input_var.shape[0], input_var.shape[1] + 10, input_var.shape[2], input_var.shape[3])).fill_(0.).cuda()
        
        adjusted_input[:,0:3,:,:] = input_var
        
        for j in range(target_var.shape[0]):
            adjusted_input[j,target_var[j]+3,:,:] = 1.
        

        # compute output
        filtered = filter_model(adjusted_input)
        output = model(filtered)
        
        confidence, prediction = output.max(1)
        
        for j in range(prediction.shape[0]):
            confusion_matrix[target[j], prediction[j]] += 1
        
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data) # [0], input.size(0))
        top1.update(prec1) # [0] , input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))
        
        noise = torch.FloatTensor(input_var.shape).uniform_(-noise_limit, noise_limit).cuda()

        # compute output
        output = model(input_var + noise)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        precn = accuracy(output.data, target)[0]
        losses.update(loss.data) # [0], input.size(0))
        topn.update(precn) # [0] , input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@n {topn.val:.3f} ({topn.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      topn=topn))
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        precp = accuracy(output.data, target)[0]
        losses.update(loss.data) # [0], input.size(0))
        topp.update(precp) # [0] , input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@p {topp.val:.3f} ({topp.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      topp=topp))
        
        filtered_ = un_normalize(filtered.detach()).cpu()
        # filtered_ = torch.max(filtered_, torch.zeros(filtered_.shape))
        # filtered_ = torch.min(filtered_, torch.ones(filtered_.shape))
        noisy_ = un_normalize((input_var + noise).detach()).cpu()
        # noisy_ = torch.max(noisy_, torch.zeros(noisy_.shape))
        # noisy_ = torch.min(noisy_, torch.ones(noisy_.shape))
        base_ = un_normalize(input_var.detach()).cpu()
        # base_ = torch.max(base_, torch.zeros(base_.shape))
        # base_ = torch.min(base_, torch.ones(base_.shape))
        
        for j in range(target.shape[0]):
        
            filtered_image = torchvision.transforms.functional.to_pil_image(filtered_[j])
            noisy_image = torchvision.transforms.functional.to_pil_image(noisy_[j])
            base_image = torchvision.transforms.functional.to_pil_image(base_[j])
            
            im_path = 'tested_images/' + str(ii) + '/'
            
            os.makedirs(im_path, exist_ok=True)

            filtered_image.save(im_path + 'filtered.jpg')
            noisy_image.save(im_path + 'noisy.jpg')
            base_image.save(im_path + 'base.jpg')
            
            with open(im_path + 'result.txt', 'w') as f:
                if int(target[j]) != int(prediction[j]):
                    fooled.append(ii)
                # else:
                #     f.write('Not fooled\n')
                
                f.write('Filter Pred class:\t' + str(int(prediction[j])) + '\n')
                f.write('Filter Pred conf:\t' + str(float(confidence[prediction[j]])) + '\n')
                f.write('Filter True class:\t' + str(int(target[j])) + '\n')
                f.write('Filter True conf:\t' + str(float(confidence[target[j]])) + '\n')
            
            ii += 1

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    print(' * Prec@n {topn.avg:.3f}'
          .format(topn=topn))
    print(' * Prec@p {topp.avg:.3f}'
          .format(topp=topp))
    
    with open('fooled.txt', 'w') as f:
        for im in fooled:
            f.write(str(im) + '\n')
    
    print(confusion_matrix)

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def save_filter_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
