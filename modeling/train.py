import torch
import torch.nn as nn
from torch.autograd import Variable
import skimage
from skimage import io
import numpy as np
import argparse

from logger import Logger
from train_loader import get_loader
from modelGeoNet import GeoNet, EPELoss
import os
os.environ['NUMBAPRO_NVVM'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\nvvm\bin\nvvm64_33_0.dll'
os.environ['NUMBAPRO_LIBDEVICE'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\nvvm\libdevice'
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='GeoNet')
parser.add_argument('--epochs', type=int, default=6, metavar='N')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
parser.add_argument('--batch_size', type=int, default=32, metavar='N')
parser.add_argument("--dataset_dir", type=str, default='/home/xliea/dataset_patch', help='dataset path')
parser.add_argument("--savemodel_dir", type=str, default='/home/xliea/model.pkl', help='save model path')
parser.add_argument("--savecheckpoint_dir", type=str, default='./CheckPnt', help='save check point model path')
args = parser.parse_args()


use_GPU = torch.cuda.is_available()

train_loader = get_loader(patch_dir = '%s%s' % (args.dataset_dir, '/train/patch'),
                  flow_dir = '%s%s' % (args.dataset_dir, '/train/patch_flow'), 
                  global_dir = '%s%s' % (args.dataset_dir, '/train/patch_4x'), 
                  batch_size = args.batch_size)

val_loader = get_loader(patch_dir = '%s%s' % (args.dataset_dir, '/test/patch'),
                 flow_dir = '%s%s' % (args.dataset_dir, '/test/patch_flow'), 
                 global_dir = '%s%s' % (args.dataset_dir, '/test/patch_4x'), 
                 batch_size = args.batch_size)

model = GeoNet([1, 1, 1, 1, 1])
criterion = EPELoss()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

step = 0
logger = Logger('./logs')

loss_data = []

model.train()
for epoch in range(args.epochs):
    for i, (local_img, flow_truth, global_img) in enumerate(train_loader):
         
        if use_GPU:
            local_img = local_img.cuda()
            flow_truth = flow_truth.cuda()
            global_img = global_img.cuda()
        
        local_img = Variable(local_img)
        flow_truth = Variable(flow_truth)
        global_img = Variable(global_img)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        
        flow_output = model(local_img, global_img)
        epe_loss = criterion(flow_output, flow_truth)

        loss = epe_loss

        loss.backward()
        optimizer.step()
        
        print("Epoch [%d], Iter [%d], Loss: %.4f" %(epoch + 1, i + 1, loss.data))

        #============ TensorBoard logging ============#
        step = step + 1
        
        #Log the scalar values
        info = {'loss': loss.data.cpu()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

    # Decaying Learning Rate
    if (epoch + 1) % 2 == 0:
        args.lr /= 2
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_data.append(loss.data.cpu())

    np.save(args.savecheckpoint_dir + 'loss' + '.npy',np.array(loss_data))

    # torch.save(model.state_dict(), r'C:\Users\ABADREDD\Desktop\Latest_School\EECS504\Project\xiaoyu258\Output2\CheckPnt\checkpnt2' + '_' + str(epoch+1) + '.pkl') 
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.data.cpu(),
                }, args.savecheckpoint_dir + 'Chkpt_' + 'transfer' + str(epoch+1) + '.pkl')
torch.save(model.state_dict(), args.savemodel_dir) 
torch.save({'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.data.cpu()},'completeModelAutin4.pkl')