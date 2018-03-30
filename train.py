'''
Training script for ImageNet
'''
import argparse
import os
import json
import shutil
import time
import random
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from resnet import resnet
from utils import AverageMeter, mkdir_p, accuracy

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Model options
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--width', default=1, type=float, help='widen factor')
parser.add_argument('--data', default='path to dataset', type=str)
parser.add_argument('--nthread', default=4, type=int,
                    help='number of data loading threads (default: 4)')

# Optimization options
parser.add_argument('--epochs', default=100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=256, type=int,
                    help='train batchsize (default: 256)')
parser.add_argument('--test_batch', default=200, type=int,
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', default='[30, 60, 90]', type=str, 
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', type=float, default=0.1, 
                    help='LR is multiplied by this ratio on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

def create_iterators():
    """Data loading code"""
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val_torch')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.nthread, pin_memory=torch.cuda.is_available())

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.nthread, pin_memory=torch.cuda.is_available())
    
    return train_loader, val_loader  
  
def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm.tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_description('Train phase | loss: {:.4f}, top1: {:.2f}, top5: {:.2f}'.format(
                             losses.avg, top1.avg, top5.avg))
        pbar.refresh() # to show immediately the update
        
    return (losses.avg, top1.avg)
              
def test(val_loader, model, criterion, epoch, use_cuda):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm.tqdm(val_loader)
    for batch_idx, (inputs, targets) in enumerate(pbar):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))        
        
        pbar.set_description('Valid phase | loss: {:.4f}, top1: {:.2f}, top5: {:.2f}'.format(
                             losses.avg, top1.avg, top5.avg))
        pbar.refresh() # to show immediately the update
        
    return (losses.avg, top1.avg)
      
def adjust_learning_rate(optimizer, epoch, schedule):
    global state
    if epoch in schedule:
        state['lr'] *= args.lr_decay_ratio
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    
def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
def main():
    global best_acc
    
    schedule = json.loads(args.schedule)
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    
    # load dataset
    train_loader, val_loader = create_iterators()
    
    # create model
    if args.pretrained:
        print('=> using pre-trained model')
    model = resnet(args.depth, args.width, pretrained=args.pretrained)
    model = torch.nn.DataParallel(model).cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])    

    start_epoch = 0

    # Evaluate
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.4f, Test Acc:  %.2f' % (test_loss, test_acc))
        return        
    
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, schedule)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
  
        # save model
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'optimizer' : optimizer.state_dict(),
        }, checkpoint=args.checkpoint, filename='resnet'+str(args.depth)+'_'+str(epoch+1)+'.pt7')
    
if __name__ == '__main__':
    main()
