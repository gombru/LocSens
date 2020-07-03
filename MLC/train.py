import time
import torch
import torch.nn.parallel
import glob
import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prec1_meter = AverageMeter()
    prec10_meter = AverageMeter()
    prec50_meter = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (image, target_indices, label) in enumerate(train_loader):

        batch_size = len(image)
        num_classes = 100000
        num_indices = 15

        # build target vector from target indices 
        target = torch.zeros((batch_size, num_classes), dtype=torch.float32).cuda(gpu, async=True)

        # target indices is [BS,num_indices]
        for p in range(0,batch_size): 
            target[p,target_indices[p]] = 1

        target_var = torch.autograd.Variable(target).squeeze(1)

        image_var = torch.autograd.Variable(image)
        label = label.cuda(gpu, async=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(image_var)
        loss = criterion(output, target_var)

        # measure and record loss
        loss_meter.update(loss.data.item(), image.size()[0])
        prec1, prec10, prec50 = accuracy(output.data, label, topk=(1, 10, 50))
        prec1_meter.update(prec1.data.item(), image.size()[0])
        prec10_meter.update(prec10.data.item(), image.size()[0])
        prec50_meter.update(prec50.data.item(), image.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec1 {prec1.val:.3f} ({prec1.avg:.3f})\t'
                  'Prec10 {prec10.val:.3f} ({prec10.avg:.3f})\t'
                  'Prec50 {prec50.val:.3f} ({prec50.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prec1=prec1_meter, prec10=prec10_meter, prec50=prec50_meter))

    plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['train_prec1'][plot_data['epoch']] = prec1_meter.avg
    plot_data['train_prec10'][plot_data['epoch']] = prec10_meter.avg
    plot_data['train_prec50'][plot_data['epoch']] = prec50_meter.avg

    return plot_data


def validate(val_loader, model, criterion, print_freq, plot_data, gpu):
    with torch.no_grad():

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        prec1_meter = AverageMeter()
        prec10_meter = AverageMeter()
        prec50_meter = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (image, target_indices, label) in enumerate(val_loader):

            # build target vector form target indices
            target = torch.zeros([len(target_indices), 100000], dtype=torch.float32).cuda(gpu, async=True)
            for p in range(0,len(target_indices)):
                target[p,target_indices[p]] = 1

            # target = target.cuda(gpu, async=True)
            target_var = torch.autograd.Variable(target).squeeze(1)

            image_var = torch.autograd.Variable(image)
            label = label.cuda(gpu, async=True)

            # compute output
            output = model(image_var)
            # output[output < 0] = -100
            loss = criterion(output, target_var)

            # measure and record loss
            loss_meter.update(loss.data.item(), image.size()[0])
            prec1, prec10, prec50 = accuracy(output.data, label, topk=(1, 10, 50))
            prec1_meter.update(prec1.data.item(), image.size()[0])
            prec10_meter.update(prec10.data.item(), image.size()[0])
            prec50_meter.update(prec50.data.item(), image.size()[0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec1 {prec1.val:.3f} ({prec1.avg:.3f})\t'
                      'Prec10 {prec10.val:.3f} ({prec10.avg:.3f})\t'
                      'Prec50 {prec50.val:.3f} ({prec50.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=loss_meter,
                    prec1=prec1_meter, prec10=prec10_meter, prec50=prec50_meter))

        plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
        plot_data['val_prec1'][plot_data['epoch']] = prec1_meter.avg
        plot_data['val_prec10'][plot_data['epoch']] = prec10_meter.avg
        plot_data['val_prec50'][plot_data['epoch']] = prec50_meter.avg

    return plot_data


def save_checkpoint(model, filename, prefix_len):
    print("Saving Checkpoint")
    torch.save(model.state_dict(), filename + '.pth.tar')

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
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res