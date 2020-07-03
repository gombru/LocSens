import time
import torch
import torch.nn.parallel
import glob
import os
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

def save_checkpoint(model, filename, prefix_len):
    print("Saving Checkpoint")
    torch.save(model.state_dict(), filename + '.pth.tar')

def train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu, margin, num_iters, var):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_pairs = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (img, tag, lat, lon) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        bs, num_neg, img_dim = img.shape

        # THIS ASSUMES THE 6 NEGATIVES HAVE SAME LOC AS THE POSITIVE!!
        # Sample lat and lon from a gaussian distribution with variance var
        for t_i in range(0,bs):
            mean = [lat[t_i,0], lon[t_i,0]]
            x = np.random.normal(mean, [var, var], [num_neg,2]) # [num_negx2]
            x = torch.from_numpy(x)
            x[x>1] = x[x>1] - x[x>1].floor()
            x[x<0] = x[x<0].floor().abs() + x[x<0]
            lat[t_i,:,0] = x[:,0]
            lon[t_i,:,0] = x[:,1]

        img = torch.autograd.Variable(img)
        tag = torch.autograd.Variable(tag)
        lat = torch.autograd.Variable(lat)
        lon = torch.autograd.Variable(lon)


        # compute scores for positive groups
        s_p = model(img[:,0,:], tag[:,0,:], lat[:,0,:], lon[:,0,:])
        y = torch.ones(img.size()[0]).cuda(gpu, async=True)
        # If `y == 1` then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for `y == -1`.
        correct = torch.zeros([1], dtype=torch.int32).cuda(gpu, async=True)

        for n_i in range(1,num_neg):
            s_n = model(img[:, n_i, :], tag[:, n_i, :], lat[:, n_i, :], lon[:, n_i, :])
            if n_i == 1:
                loss = criterion(s_p, s_n, y)
                # Check if pair is already correct (not used for the loss, just for monitoring)
                for batch_idx in range(0, len(s_n)):
                    if (s_p[batch_idx] - s_n[batch_idx]) > margin:
                        correct[0] += 1
            else:
                loss += criterion(s_p, s_n, y)

        loss /= (num_neg-1)

        # measure and record loss
        loss_meter.update(loss.data.item(), img.size()[0])
        correct_pairs.update(torch.sum(correct))

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
                  'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
                  'Correct Pairs {correct_pairs.val:.3f} ({correct_pairs.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, correct_pairs=correct_pairs))

        if i == num_iters-1:
            plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
            plot_data['train_correct_pairs'][plot_data['epoch']] = correct_pairs.avg
            return plot_data

    plot_data['train_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['train_correct_pairs'][plot_data['epoch']] = correct_pairs.avg


    return plot_data


def validate(val_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu, margin, num_iters, var):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_pairs = AverageMeter()

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (img, tag, lat, lon) in enumerate(val_loader):

            # measure data loading time
            data_time.update(time.time() - end)
            bs, num_neg, img_dim = img.shape

            # THIS ASSUMES THE 6 NEGATIVES HAVE SAME LOC AS THE POSSITIVE!!
            # Sample lat and lon from a gaussian distribution with variance var
            
            # for t_i in range(0,bs):
            #     mean = [lat[t_i,0], lon[t_i,0]]
            #     x = np.random.normal(mean, [var, var], [num_neg,2]) # [num_negx2]
            #     x = torch.from_numpy(x)
            #     x[x>1] = x[x>1] - x[x>1].floor()
            #     x[x<0] = x[x<0].floor().abs() + x[x<0]
            #     lat[t_i,:,0] = x[:,0]
            #     lon[t_i,:,0] = x[:,1]

            img = torch.autograd.Variable(img)
            tag = torch.autograd.Variable(tag)
            lat = torch.autograd.Variable(lat)
            lon = torch.autograd.Variable(lon)

            # compute scores for positive groups
            s_p = model(img[:, 0, :], tag[:, 0, :], lat[:, 0, :], lon[:, 0, :])

            y = torch.ones(img.size()[0]).cuda(gpu, async=True)
            # If `y == 1` then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for `y == -1`.
            correct = torch.zeros([1], dtype=torch.int32).cuda(gpu, async=True)

            for n_i in range(1, num_neg):
                s_n = model(img[:, n_i, :], tag[:, n_i, :], lat[:, n_i, :], lon[:, n_i, :])
                if n_i == 1:
                    loss = criterion(s_p, s_n, y)
                    # Check if pair is already correct (not used for the loss, just for monitoring)
                    for batch_idx in range(0, len(s_n)):
                        if (s_p[batch_idx] - s_n[batch_idx]) > margin:
                            correct[0] += 1
                else:
                    loss += criterion(s_p, s_n, y)


            loss /= (num_neg-1)

            # measure and record loss
            loss_meter.update(loss.data.item(), img.size()[0])
            correct_pairs.update(torch.sum(correct))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % print_freq == 0:
                print('Validation: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.7f} ({loss.avg:.7f})\t'
                      'Correct Pairs {correct_pairs.val:.3f} ({correct_pairs.avg:.3f})\t'
                      .format(
                       epoch, i, len(val_loader), batch_time=batch_time,
                       data_time=data_time, loss=loss_meter, correct_pairs=correct_pairs))

            if i == num_iters-1:
                plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
                plot_data['val_correct_pairs'][plot_data['epoch']] = correct_pairs.avg
                return plot_data

    plot_data['val_loss'][plot_data['epoch']] = loss_meter.avg
    plot_data['val_correct_pairs'][plot_data['epoch']] = correct_pairs.avg


    return plot_data



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