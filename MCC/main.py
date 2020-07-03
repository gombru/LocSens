# Train MCC model

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import YFCC_dataset
import train
import model
from pylab import zeros, arange, subplots, plt, savefig

# Config
training_id = 'YFCC_MCC'
dataset = '../../../hd/datasets/YFCC100M/'
split_train = 'train_filtered.txt'
split_val = 'val.txt'

ImgSize = 224
gpus = [3,2,1] # [3,2,1,0]
gpu = 3
workers = 10 # 6 Num of data loading workerss
epochs = 301
start_epoch = 0 # Useful on restarts
batch_size = 96 * len(gpus) # Batch size
print_freq = 1 # An epoch are 60000 iterations. Print every 100: Every 40k images
resume = None  # Path to checkpoint top resume training
plot = True
best_epoch = 0
best_loss = 1000

# Optimizer (SGD)
lr = 2*1e-2 * len(gpus)
momentum = 0.9
weight_decay = 1e-4

# Loss
criterion = nn.CrossEntropyLoss().cuda(gpu)
# Model
model = model.Model().cuda(gpu)
model = torch.nn.DataParallel(model, device_ids=gpus)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


# Optionally resume from a checkpoint
if resume:
    print("Loading pretrained model")
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})
    model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded")

cudnn.benchmark = True

# Data loading code (pin_memory allows better transferring of samples to GPU memory)
train_dataset = YFCC_dataset.YFCC_Dataset(
    dataset,split_train,random_crop=ImgSize,mirror=True)

val_dataset = YFCC_dataset.YFCC_Dataset(
    dataset, split_val,random_crop=ImgSize,mirror=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

# Plotting config
plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['train_prec1'] = zeros(epochs)
plot_data['train_prec5'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['val_prec1'] = zeros(epochs)
plot_data['val_prec5'] = zeros(epochs)
plot_data['epoch'] = 0
it_axes = arange(epochs)
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss (r), val loss (y)')
ax2.set_ylabel('train prec1 (b), train prec5 (k), val prec1 (g), val prec5 (m)')
ax2.set_autoscaley_on(False)
ax1.set_ylim([0, 12])
ax2.set_ylim([0, 100])

print("Dataset and model ready. Starting training ...")

for epoch in range(start_epoch, epochs):
    plot_data['epoch'] = epoch

    # Train for one epoch
    plot_data = train.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu)

    # Evaluate on validation set
    plot_data = train.validate(val_loader, model, criterion, print_freq, plot_data, gpu)

    # Remember best model and save checkpoint
    is_best = plot_data['val_loss'][epoch] < best_loss
    if is_best and epoch != 0:
        print("New best model by loss. Val Loss = " + str(plot_data['val_loss'][epoch]))
        best_loss = plot_data['val_loss'][epoch]
        filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2))
        prefix_len = len('_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2)))
        train.save_checkpoint(model, filename, prefix_len)

    if plot:
        ax1.plot(it_axes[0:epoch], plot_data['train_loss'][0:epoch], 'r')
        ax2.plot(it_axes[0:epoch], plot_data['train_prec1'][0:epoch], 'b')
        ax2.plot(it_axes[0:epoch], plot_data['train_prec5'][0:epoch], 'k')


        ax1.plot(it_axes[0:epoch], plot_data['val_loss'][0:epoch], 'y')
        ax2.plot(it_axes[0:epoch], plot_data['val_prec1'][0:epoch], 'g')
        ax2.plot(it_axes[0:epoch], plot_data['val_prec5'][0:epoch], 'm')
        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

        # Save graph to disk
        if epoch % 1 == 0 and epoch != 0:
            title = dataset +'/training/' + training_id + '.png'
            savefig(title, bbox_inches='tight')

