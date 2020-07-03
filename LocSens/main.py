import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import YFCC_dataset as CustomDataset
import train
import model
from pylab import zeros, arange, subplots, plt, savefig

# Config
training_id = 'geoModel_retrieval_ProgressiveFusionGaussianSampling'

dataset = '../../../hd/datasets/YFCC100M/'
split_train = 'train_filtered.txt'
split_val = 'val.txt'

img_backbone_model = 'YFCC_MCC'

margin = 0.1

gpus = [3]
gpu = 3
workers = 0 # 8 Num of data loading workers
epochs = 2000
start_epoch = 0 # Useful on restarts
batch_size = 1024 # * 10 # 1024 # Batch size
print_freq = 1 # An epoch are 60000 iterations. Print every 100: Every 40k images
resume = dataset + 'models/geoModel_retrieval_ProgressiveFusionGaussianSampling.pth.tar'  # Path to checkpoint top resume training
plot = True 
best_epoch = 0
best_correct_pairs = 0
best_loss = 1000

train_iters = 0
val_iters = 0

# Optimizer (SGD)
lr = 0.0001
momentum = 0.9
weight_decay = 1e-4

variance = 0
variance_step = 0.001

# Loss
criterion = nn.MarginRankingLoss(margin=margin).cuda(gpu)
# Model
model = model.Model_Multiple_Negatives().cuda(gpu)
model = torch.nn.DataParallel(model, device_ids=gpus)

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Optionally resume from a checkpoint
if resume:
    print("Loading pretrained model")
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location={'cuda:0':'cuda:3', 'cuda:2':'cuda:3', 'cuda:2':'cuda:3'})
    model.load_state_dict(checkpoint, strict=False)
    print("Checkpoint loaded")

cudnn.benchmark = True

# Data loading code (pin_memory allows better transferring of samples to GPU memory)
train_dataset = CustomDataset.YFCC_Dataset(
    dataset,split_train,img_backbone_model)

val_dataset = CustomDataset.YFCC_Dataset(
    dataset, split_val,img_backbone_model)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

# Plotting config
plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['train_correct_pairs'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['val_correct_pairs'] = zeros(epochs)
plot_data['epoch'] = 0
it_axes = arange(epochs)
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss (r), val loss (y)')
ax2.set_ylabel('train correct pairs (b), val correct pairs (g)')
ax2.set_autoscaley_on(False)
ax1.set_ylim([0.015, 0.03])
ax2.set_ylim([700, batch_size + 0.2])

print("Dataset and model ready. Starting training ...")

cur_var_not_best = 0

for epoch in range(start_epoch, epochs):

    plot_data['epoch'] = epoch

    # Train for one epoch
    plot_data = train.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu, margin, train_iters, variance)

    # Evaluate on validation set
    plot_data = train.validate(val_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu, margin, val_iters, variance)

    # Remember best model and save checkpoint
    is_best = plot_data['val_loss'][epoch] < best_loss
    if is_best:
        best_model = 1
    else:
        best_model = 0
        cur_var_not_best+=1

    if is_best:
        print("New best model by loss. Val Loss = " + str(plot_data['val_loss'][epoch]))
        best_loss = plot_data['val_loss'][epoch]
        filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],3)) + '_var_' + str(round(variance,4))
        prefix_len = len('_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],3)))
        train.save_checkpoint(model, filename, prefix_len)

    elif epoch % 1 == 0:
        filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch) + '_ValLossNotBest' + str(round(plot_data['val_loss'][epoch],3)) + '_var_' + str(round(variance,4))
        prefix_len = len('_epoch_' + str(epoch) + '_ValLossNotBest_' + str(round(plot_data['val_loss'][epoch],3)))
        train.save_checkpoint(model, filename, prefix_len)


    if plot:
        ax1.plot(it_axes[0:epoch+1], plot_data['train_loss'][0:epoch+1], 'r')
        ax2.plot(it_axes[0:epoch+1], plot_data['train_correct_pairs'][0:epoch+1], 'b')

        ax1.plot(it_axes[0:epoch+1], plot_data['val_loss'][0:epoch+1], 'y')
        ax2.plot(it_axes[0:epoch+1], plot_data['val_correct_pairs'][0:epoch+1], 'g')

        plt.title(training_id + str(round(variance,4)), fontsize=10)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

        # Save graph to disk
        if epoch % 1 == 0 and epoch != 0:
            title = dataset +'/training/' + training_id + '_epoch_' + str(epoch) + '_var_' + str(round(variance,4)) + '.png'
            savefig(title, bbox_inches='tight')

    variance+=variance_step

print("Finished Training, saving checkpoint")
filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch)
prefix_len = len('_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2)))
train.save_checkpoint(model, filename, prefix_len)

