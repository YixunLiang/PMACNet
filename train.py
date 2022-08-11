# coding=gbk
import numpy as np
from config import TrainGlobalConfig
config = TrainGlobalConfig
import torch.backends.cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from os.path import join
import random
from model import *
from osgeo import gdal
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_id
torch.cuda.current_device()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
mav_value = 2047.  #归一化50

# 文件夹设置
def set_ckpt(method,satellite):
    traindata_dir = 'datasets/{}/train/'.format(satellite)  # 训练数据路径
    validdata_dir = 'datasets/{}/valid/'.format(satellite)  # 验证数据路径
    trainrecord_dir = './ckpts/models-{}-{}/'.format(method, satellite)  # 训练loss记录路径
    validrecord_dir = './ckpts/models-{}-{}/'.format(method, satellite)  # 训练loss记录路径
    model_dir = './ckpts/models-{}-{}/'.format(method, satellite)  # 模型保存路径
    checkpoint_model = join(model_dir, '{}.pth'.format(method))
    return{
        'traindata_dir':traindata_dir,
        'validdata_dir':validdata_dir,
        'trainrecord_dir':trainrecord_dir,
        'validrecord_dir':validrecord_dir,
        'model_dir':model_dir,
        'checkpoint_model':checkpoint_model
    }

ckpt_dir = set_ckpt(config.method,config.satellite)

if not os.path.exists(ckpt_dir['trainrecord_dir']):
    os.makedirs(ckpt_dir['trainrecord_dir'])
if not os.path.exists(ckpt_dir['validrecord_dir']):
    os.makedirs(ckpt_dir['validrecord_dir'])
if not os.path.exists(ckpt_dir['model_dir']):
    os.makedirs(ckpt_dir['model_dir'])


# 配置gpu
device = torch.device('cuda')
SEED = config.seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)
def load_image(filepath):
    '''
    读取卫星图像
    :param filepath:
    :return:
    '''
    dataset = gdal.Open(filepath)
    img = dataset.ReadAsArray()
    img = img.astype(np.float32) / mav_value
    return img


class DatasetFromFolder(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = [x for x in os.listdir(img_dir+'MS')]
        random.shuffle(self.image_filenames)

    def __len__(self):
        return len(self.image_filenames) #len(self.image_filenames)

    def __getitem__(self, index):  # idx的范围是从0到len（self）
        ms = load_image(self.img_dir+'MS/{}'.format(self.image_filenames[index]))
        pan = load_image(self.img_dir + 'PAN/{}'.format(self.image_filenames[index]))
        gt = load_image(self.img_dir+'GT/{}'.format(self.image_filenames[index]))

        if self.transform:
            ms = self.transform(ms)
            pan = self.transform(pan)
            gt = self.transform(gt)
        return ms,pan,gt


# 将矩阵变换为Tensor
class ToTensor(object):
    def __call__(self, input):
        if input.ndim == 3:
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input


def get_train_set(traindata_dir):
    return DatasetFromFolder(traindata_dir,
                             transform=transforms.Compose([ToTensor()]))


def get_valid_set(validdata_dir):
    return DatasetFromFolder(validdata_dir,
                             transform=transforms.Compose([ToTensor()]))


transformed_trainset = get_train_set(ckpt_dir['traindata_dir'])
transformed_validset = get_valid_set(ckpt_dir['validdata_dir'])
# print('train:', len(transformed_trainset))
# print('valid:', len(transformed_validset))

# 训练集设置
trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=config.train_batch_size, shuffle=True,
                                 num_workers=config.num_workers, pin_memory=True, drop_last=True)
validset_dataloader = DataLoader(dataset=transformed_validset, batch_size=config.valid_batch_size, shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True, drop_last=True)


# 取下一个batch
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream(device).wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch



def adjust_learning_rate(lr, epoch, freq):
    lr = lr * (0.5 ** (epoch // freq))
    return lr


# Device setting
criterion = nn.MSELoss().to(device)
model = PMACNet(ms_inp_ch=config.ms_inp_ch,num_layers=config.num_layers,latent_dim=config.latent_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)


# 模型训练
total_epochs = config.total_epochs
def train(model, trainset_dataloader, start_epoch):
    print('===>Begin Training!')
    train_steps_per_epoch = len(trainset_dataloader)
    valid_steps_per_epoch = len(validset_dataloader)
    total_iterations = total_epochs * train_steps_per_epoch
    print('total_iterations:{}'.format(total_iterations))
    train_loss_record = open('%s/train_loss_record.txt' % ckpt_dir['trainrecord_dir'], "w")
    valid_loss_record = open('%s/valid_loss_record.txt' % ckpt_dir['validrecord_dir'], "w")

    for epoch in range(start_epoch + 1, total_epochs + 1):
        # 配置更新的学习率
        learning_rate = adjust_learning_rate(config.lr, epoch - 1, config.lr_decay_freq)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        print("=>epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])


        # 训练网络
        train_prefetcher = DataPrefetcher(trainset_dataloader)
        train_data = train_prefetcher.next()
        train_total_loss = 0

        model.train()
        for batch in range(train_steps_per_epoch):
            img_ms,img_pan, gt = train_data[0], train_data[1], train_data[2]
            img_ms =F.interpolate(img_ms,scale_factor=4,mode='bicubic',align_corners=True).to(device)
            img_pan = img_pan.to(device)
            gt = gt.to(device)
#这里把ms与pan图叠起来，高通送入模型
#loss function为pred - （GT-M上采样）
            #注意！！！！ms4loss是不是这么写
            pred = model(img_ms,img_pan)
            train_loss = criterion(pred+img_ms, gt)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_data = train_prefetcher.next()# 取下一个batch数据
            train_total_loss += train_loss.item()

        #验证网络
        valid_prefetcher = DataPrefetcher(validset_dataloader)
        valid_data = valid_prefetcher.next()
        valid_total_loss = 0
        model.eval()
        for batch in range(valid_steps_per_epoch):
            img_ms,img_pan, gt = valid_data[0], valid_data[1],valid_data[2]
            img_ms =F.interpolate(img_ms,scale_factor=4,mode='bicubic',align_corners=False).to(device)
            img_pan = img_pan.to(device)
            gt = gt.to(device)
            pred = model(img_ms,img_pan)
            valid_loss = criterion(pred+img_ms, gt)
            valid_data = valid_prefetcher.next()
            valid_total_loss += valid_loss.item()
        # 保存模型参数
        train_avg_loss = train_total_loss/train_steps_per_epoch
        valid_avg_loss = valid_total_loss / valid_steps_per_epoch
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
        }
        torch.save(state,ckpt_dir['checkpoint_model'])
        # 记录loss值
        print('=>Epoch[{}/{}]: train_loss: {:.15f} valid_loss: {:.15f}'.format(epoch, total_epochs, train_avg_loss, valid_avg_loss))
        train_loss_record.write(
            "Epoch[{}/{}]: train_loss: {:.15f}\n".format(epoch, total_epochs, train_avg_loss))
        valid_loss_record.write(
            "Epoch[{}/{}]: valid_loss: {:.15f}\n".format(epoch, total_epochs, valid_avg_loss))

    train_loss_record.close()
    valid_loss_record.close()


def main():
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(ckpt_dir['checkpoint_model']) and config.resume == True:
        print("==> loading checkpoint '{}'".format(ckpt_dir['checkpoint_model']))
        checkpoint = torch.load(ckpt_dir['checkpoint_model'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')

    train(model, trainset_dataloader, start_epoch)


if __name__ == '__main__':
    main()
