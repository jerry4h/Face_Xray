from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from core.evaluate import accuracy_classify

def train(config, train_loader, nnb, nnc, criterion, optimizer, epoch, writer_dict, _print):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    lossNNB = AverageMeter()
    lossNNC = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    nnb.train()
    nnc.train()

    end = time.time()
    for i, (input, target, cls_target, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # 非阻塞允许多个线程同时进入临界区
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        cls_target = cls_target.cuda(non_blocking=True)

        # compute output
        output = nnb(input)
        classify = nnc(output)

        loss_nnb = criterion.facexray_loss(output, target)
        loss_nnc = criterion.nnc_loss(classify, cls_target)
        loss = loss_nnb * 100 + loss_nnc

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        lossNNB.update(loss_nnb.item(), input.size(0))
        lossNNC.update(loss_nnc.item(), input.size(0))

        # nnc 的输出
        acc = accuracy_classify(classify, cls_target) * 100

        accs.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.1f}s ({batch_time.avg:.1f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.1f}s ({data_time.avg:.1f}s)\t' \
                  'Loss {loss.val:.2f} ({loss.avg:.2f})\t' \
                  'LossNNB {lossNNB.val:.2f} ({lossNNB.avg:.2f})\t' \
                  'LossNNC {lossNNC.val:.2f} ({lossNNC.avg:.2f})\t' \
                  'Acc {accuracy.val:.1f} ({accuracy.avg:.1f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses,
                      lossNNB=lossNNB, lossNNC=lossNNC, accuracy=accs)
            _print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', accs.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

def validate(config, val_loader, nnb, nnc, criterion, writer_dict, _print, isTrain=False):
    batch_time = AverageMeter()
    lossNNB = AverageMeter()
    lossNNC = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluate mode
    nnb.eval()
    nnc.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, cls_target, _, _) in enumerate(val_loader):

            # 非阻塞允许多个线程同时进入临界区
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            cls_target = cls_target.cuda(non_blocking=True)

            # compute output
            output = nnb(input)
            classify = nnc(output)

            # compute loss
            loss_nnb = criterion.facexray_loss(output, target)
            loss_nnc = criterion.nnc_loss(classify, cls_target)
            loss = loss_nnb * 100 + loss_nnc

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            lossNNB.update(loss_nnb.item(), input.size(0))
            lossNNC.update(loss_nnc.item(), input.size(0))

            acc = accuracy_classify(classify, cls_target, isTrain) * 100

            accs.update(acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.1f}\t' \
              'Loss {loss.avg:.2f}\t' \
              'LossNNB {lossnnb.avg:.2f}\t' \
              'LossNNC {lossnnc.avg:.2f}\t' \
              'Accuracy {accuracy.avg:.2f}\t'.format(
                  batch_time=batch_time, loss=losses, lossnnb=lossNNB, lossnnc=lossNNC,
                   accuracy=accs)
        _print(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', accs.avg, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return accs.val



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
