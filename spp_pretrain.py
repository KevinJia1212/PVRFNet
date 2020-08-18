import sys
sys.path.append('/home/aistudio/external-libraries')
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from termcolor import colored

from scipy.spatial.distance import cdist
# from original_model import Net
# from original_model64 import Net
from model_spp2 import PVRFNet_SPP
from utils import market1501, veri776, util, eval_tools, fused_dataset, triplet, sampler

parser = argparse.ArgumentParser(description="Train on market1501 and veri776")
parser.add_argument("--market_dir",default='data',type=str)
parser.add_argument("--veri_dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.001, type=float)
parser.add_argument("--lr_step",default=0.3, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--image_size', default=64, type=int, help='input image size of the network')
parser.add_argument('--batch_size', default=1600, type=int, help='Batch size for training')
parser.add_argument('--ids_minibatch', default=4, type=int, help='pic num of each id in one batch')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from.')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--margin', default=1.2, type=float)
parser.add_argument('--epoches', default=100, type=float)
args = parser.parse_args()

#### 训Triplet需要大batch_size，小ids_minibatch ###

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# transform defination
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.image_size,args.image_size)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.image_size,args.image_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# data loading
market_root = args.market_dir
veri_root = args.veri_dir

data = fused_dataset.Fused_Dataset(market_root, veri_root, transform_train, transform_test)

trainloader = torch.utils.data.DataLoader(data.train, batch_size=args.batch_size, sampler=sampler.RandomIdentitySampler(data.train, args.ids_minibatch), num_workers=args.num_workers)
#trainloader = torch.utils.data.DataLoader(data.train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(data.test, batch_size=1024, num_workers=8)
queryloader = torch.utils.data.DataLoader(data.query, batch_size=1024, num_workers=8)



# net definition
num_classes = len(np.unique(data.train.ids))
start_epoch = 0
start_lr = args.lr
lr_adjust_list = [60,90,160,180]
#lr_adjust_list = [240]
net = PVRFNet_SPP(num_classes=num_classes)
if args.resume is not None:
    assert os.path.isfile(args.resume), "Error: no checkpoint file found!"
    print('Loading from {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    # best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    new_lr_adjust_list = []
    for step in lr_adjust_list:
        if start_epoch >= step:
            start_lr *= args.lr_step 
        else:
            new_lr_adjust_list.append(step)
    for i in range(len(new_lr_adjust_list)):
        new_lr_adjust_list[i] -= start_epoch
    lr_adjust_list = new_lr_adjust_list

net.to(device)

# loss and optimizer
ce_loss = torch.nn.CrossEntropyLoss()
trp_loss = triplet.TripletSemihardLoss(args.margin)
# trp2_loss = triplet.TripletLoss(args.margin)
optimizer = torch.optim.Adam(net.parameters(), lr=start_lr, betas=(0.9, 0.99), weight_decay=0.01)
# optimizer = torch.optim.SGD(net.parameters(), start_lr, momentum=0.9, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_adjust_list, gamma=args.lr_step)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("Epoch : %d"%(epoch+1))
    net.is_train = True
    net.train()
    training_loss = 0.
    iding_loss = 0.
    triing_loss = 0.
    train_loss = 0.
    correct = 0
    precision = 0.
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device),labels.to(device)
        # print(np.unique(np.asarray(labels.cpu())))
        features, classes = net(inputs)
        id_loss = ce_loss(classes, labels)
        tri_loss, prec = trp_loss(features, labels)
        loss = id_loss + tri_loss
        # loss = id_loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        train_loss += loss.item()
        iding_loss += id_loss
        triing_loss += tri_loss
        correct += classes.max(dim=1)[1].eq(labels).sum().item()
        precision += prec
        total += labels.size(0)


        # print 
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s TotalLoss:{:.5f} id_loss:{:.5f} tri_loss:{:.5f} Correct:{}/{} Acc:[{:.3f}%] Prec:[{:.3f}%] lr:{:.2g}".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, iding_loss/interval, triing_loss/interval, correct, total, 100.*correct/total, 100.*precision/interval, optimizer.param_groups[0]['lr']
            ))
            # print("[progress:{:.1f}%]time:{:.2f}s TotalLoss:{:.5f} id_loss:{:.5f} Correct:{}/{} Acc:[{:.3f}%]  lr:{:.2g}".format(
            #     100.*(idx+1)/len(trainloader), end-start, training_loss/interval, iding_loss/interval,  correct, total, 100.*correct/total,  optimizer.param_groups[0]['lr']
            # ))
            training_loss = 0.
            iding_loss = 0.
            triing_loss = 0.
            precision = 0.
            start = time.time()
    
    return train_loss/len(trainloader)

# # lr decay
# def lr_decay():
#     global optimizer
#     for params in optimizer.param_groups:
#         params['lr'] *= 0.1
#         lr = params['lr']
#         print("Learning rate adjusted to {}".format(lr))

def eval(epoch):
    net.is_train = False
    net.eval()
    query = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in queryloader])
    test = np.concatenate([net(inputs.to(device)).detach().cpu().numpy() for inputs, _ in testloader])
    dist = cdist(query, test)

    r = eval_tools.cmc(dist, data.query.ids, data.test.ids, data.query.cameras, data.test.cameras,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True,
            same_cam_valid=True)

    m_ap = eval_tools.mean_ap(dist, data.query.ids, data.test.ids, data.query.cameras, data.test.cameras, same_cam_valid=True)
    print(colored('epoch[%d]: mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (epoch + 1, m_ap, r[0], r[2], r[4], r[9]), "yellow"))


def main():
    try:
        for epoch in range(start_epoch, start_epoch+args.epoches):
            train_loss = train(epoch)
            scheduler.step()
            # test_loss, test_err = test(epoch)
            if (epoch+1) % 10 == 0:
                eval(epoch)
            if (epoch+1) % 50 == 0:
                print("Saving parameters to checkpoint/")
                checkpoint = {
                    'net_dict':net.state_dict(),
                    'epoch':epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                ckpt_path = "checkpoint/spp2_ckpt_" + str(epoch) + ".t7" 
                torch.save(checkpoint, ckpt_path)
            # draw_curve(epoch, train_loss, train_err, test_loss, test_err)
            # if (epoch+1)%10==0:
            #     lr_decay()
    except KeyboardInterrupt:
        print("Stop early. Saving checkpoint")
        checkpoint = {
                    'net_dict':net.state_dict(),
                    'epoch':epoch,
                }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        ckpt_path = "checkpoint/spp2_ckpt_" + str(epoch) + ".t7" 
        torch.save(checkpoint, ckpt_path)
        


if __name__ == '__main__':
    main()
