'''Train CIFAR10 with PyTorch.'''
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import copy
import argparse
import tqdm

from models import *
from models.densenet2 import *

from efficientnet_pytorch import EfficientNet

def get_lr(optimizer):
    for p in optimizer.param_groups:
        break
    return p['lr']

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', type=int, default=64, help='batch_size')
parser.add_argument('--epochs', type=int, default=400, help='epochs')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--model', type=str, default='DenseNet3')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--layers', type=int, default=40)
parser.add_argument('--bottleneck', action='store_true')

args = parser.parse_args()

data_root = '/home/zliangak/data/'
start_epoch = 0

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def modify_state_keys(logs):
    keys = copy.deepcopy(list(logs['best_model'].keys()))
    for k in keys:
        logs['best_model'][k.replace('module.', '')] = logs['best_model'][k]
        del logs['best_model'][k]
    return logs

# Data
print('==> Preparing data..')

if 'efficientnet' in args.model:
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

else:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = torchvision.datasets.CIFAR10(root=data_root+'data_Cifar10', train=True, download=True, transform=transform_train)
trainset.data = trainset.data[:45000]
trainset.targets = trainset.targets[:45000]
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root=data_root+'data_Cifar10', train=True, download=True, transform=transform_train)
valset.data = valset.data[45000:]
valset.targets = valset.targets[45000:]
valloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_root+'data_Cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logs = {'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': 0,
        'best_val': 0,
        'test_acc': 0,
        'best_model': None,
        'last_model': None,
        'arguments': None,
        'state_dict': [],
        'param_size': 0,
        'lr':[]
       }

logs['arguments'] = str(args)

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()

if 'efficientnet' in args.model:
    if args.pretrained:
        print('==> Loading pretrained model')
        #net = EfficientNet.from_pretrained(args.model)
        net = EfficientNet.from_pretrained(args.model, num_classes=10)
    else:
        #net = EfficientNet.from_name(args.model)
        net = EfficientNet.from_name(args.model, {'num_classes':10})
elif 'DenseNet3' in args.model:
    net = DenseNet3(args.layers, 10, 12, reduction=1, bottleneck=args.bottleneck, dropRate=0)
    nesterow = True
else:
    net = eval(args.model+'()')

if args.resume:
    print('==> Resuming from checkpoint')
    assert os.path.exists('../weights/'+args.resume), 'Error: No such checkpoint'
    logs = torch.load('../weights/'+args.resume, map_location=torch.device('cpu'))
    try:
        net.load_state_dict(logs['best_model'])
    except:
        net.load_state_dict(modify_state_keys(logs)['best_model'])
    start_epoch = len(logs['train_loss'])

net = net.cuda()
logs['param_size'] = sum([p.numel() for p in net.parameters() if p.requires_grad])
print(args)
print("Model size %.5f M"%(logs['param_size']/1000000))

#net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.optim == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
elif args.optim == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise 'optim should in [sgd, adam]'

schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)

def test(net, loader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 1.*correct/total
    loss =  test_loss/(batch_idx+1)

    return acc, loss

try:

    for epoch in range(start_epoch, start_epoch+args.epochs):
        net.train()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader),
                                                      desc='Epoch: %d/%d, lr :%5f'%(epoch, args.epochs+start_epoch, get_lr(optimizer)), total=len(trainloader)):
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        logs['train_loss'].append(train_loss/(batch_idx+1))
        logs['train_acc'].append(1.*correct/total)

        val_acc, val_loss = test(net, valloader)
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)

        logs['lr'].append(get_lr(optimizer))

        if logs['best_val'] < val_acc:
            logs['best_val'] = val_acc
            logs['best_model'] = copy.deepcopy(net.state_dict())

        print('Train Loss {:.4f} --- Val Loss {:.4f}'.format(train_loss/(batch_idx+1), val_loss))
        print("Train acc {:.4f} --- Val acc {:.4f} -- Best acc {:.4f}".format(1.*correct/total, val_acc, logs['best_val']))

        if 'DenseNet3' in args.model:
            adjust_learning_rate(optimizer, epoch)
        else:
            schedule.step()

    net.load_state_dict(logs['best_model'])
    logs['test_acc'], logs['test_loss'] = test(net, testloader)
    print("---------------------------------------------------------------")
    print('Test acc {:.4f} --- Test loss {:.4f}'.format(logs['test_acc'], logs['test_loss']))

    torch.save(logs, '../weights/{}_size_{:4f}_pretrain_{}_epochs_{}_lr_{:5f}_bs_{}_valacc_{:4f}_testacc_{:4f}'.format(
        args.model, logs['param_size'] / 1000000, str(args.pretrained), epoch, args.lr, args.bs, logs['best_val'], logs['test_acc']))

except KeyboardInterrupt:
    torch.save(logs, '../weights/{}_size_{:4f}_pretrain_{}_epochs_{}_lr_{:5f}_bs_{}_valacc_{:4f}_testacc_{:4f}'.format(
        args.model, logs['param_size'] / 1000000, str(args.pretrained), epoch, args.lr, args.bs, logs['best_val'], logs['test_acc']))
