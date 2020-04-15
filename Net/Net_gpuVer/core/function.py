from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from core.evaluate import accuracy

def train(config, train_loader, model, criterion, optimizer, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # 非阻塞允许多个线程同时进入临界区
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)

        loss = criterion.facexray_loss(output, target)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # 改用语义分割的PA准确率
        acc,_ = accuracy(output, target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {accuracy:.4f}\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, accuracy=acc)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

def validate(config, val_loader, model, criterion, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, _, _) in enumerate(val_loader):

            # 非阻塞允许多个线程同时进入临界区
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)

            loss = criterion.facexray_loss(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            acc, _ = accuracy(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        msg = 'Test: Time {batch_time.avg:.3f}\t' \
              'Loss {loss.avg:.4f}\t' \
              'Accuracy {accuracy:.4f}\t'.format(
                  batch_time=batch_time, loss=losses, accuracy=acc)
        print(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1



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
