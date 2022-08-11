
import numpy as np
import torch
import cv2
from osgeo import gdal
from config import TrainGlobalConfig
config = TrainGlobalConfig
from model import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_id
# 将矩阵变换为Tensor
class ToTensor(object):
    def __call__(self, input):
        if input.ndim == 3:
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input


device = torch.device('cuda:0')
max_value = 2047.  #归一化
satellite = config.satellite
method = config.method
data_dir = 'datasets/{}/test/'.format(satellite)  # 验证数据路径
img_save_dir = './gen_imgs/-{}-{}/' .format(method, satellite)#测试数据路径
model = PMACNet(ms_inp_ch=config.ms_inp_ch,num_layers=config.num_layers,latent_dim=config.latent_dim).to(device)
model.load_state_dict(torch.load("./ckpts/models-{}-{}/{}.pth".format(method, satellite, method),map_location='cuda:0')['model_state_dict'])
model.eval()

if not os.path.exists(img_save_dir+"/RR-{}-Fused-{}/".format(method, satellite)):
    os.makedirs(img_save_dir+"/RR-{}-Fused-{}/".format(method, satellite))

if not os.path.exists(img_save_dir+"/FR-{}-Fused-{}/".format(method, satellite)):
    os.makedirs(img_save_dir+"/FR-{}-Fused-{}/".format(method, satellite))

def load_image(filepath):
    '''
    读取卫星图像
    :param filepath:
    :return:
    '''
    dataset = gdal.Open(filepath)
    img = dataset.ReadAsArray()
    img = img.astype(np.float32) / max_value
    return img

def saveimg(filename, im_data):
    im_data = np.uint16(im_data)
    datatype = gdal.GDT_UInt16
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, (2, 0, 1))
    else:
        im_data = np.expand_dims(im_data, axis=0)
    im_bands, im_height, im_width = im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset


def show16(windowname, img, bitnum=11):
    '''
    显示指定位数的图像，一般卫星图像为11位
    :param windowname:
    :param img:
    :param bitnum:
    :return:
    '''
    img = np.uint16(img)
    if len(img.shape) == 3:
        w, h, bands = img.shape
    else:
        w, h = img.shape
        bands = 1
    min, max = np.zeros(bands), np.zeros(bands)
    for i in range(bands):
        hist = cv2.calcHist([img], [i], None, histSize=[2**bitnum], ranges=[0, 2**bitnum-1])
        p = 0
        while p < 0.02 and min[i] <= 2**bitnum:
            p += hist[int(min[i])]/(w*h)
            min[i] = min[i]+1
        p = 0
        while p < 0.98 and max[i] <= 2**bitnum:
            p += hist[int(max[i])]/(w*h)
            max[i] = max[i]+1
    img = np.clip((img.astype(np.float32)-min)/(max-min), 0, 1)
    cv2.imshow(windowname, img)


def forward(model, pan, ms):
    '''
    模型前向传播得到融合图像
    :param model:
    :param pan:
    :param ms:
    :return:
    '''
    pan = np.expand_dims(pan, axis=0)
    ms = torch.from_numpy(ms).unsqueeze(0).type(torch.FloatTensor).to(device)
    pan = torch.from_numpy(pan).unsqueeze(0).type(torch.FloatTensor).to(device)
    img_ms = F.interpolate(ms, scale_factor=4, mode='bicubic', align_corners=True).to(device)
    result = model(img_ms,pan)
    result = result+img_ms
    result = result.cpu()
    result = result.numpy()[0]
    result = np.transpose(result, (1, 2, 0))
    return result


length = 100
with torch.no_grad():
    for i in range(length):
        pan = load_image(data_dir+"Reduced Resolution/PAN/"+str(i+1)+".TIF")
        ms = load_image(data_dir+"Reduced Resolution/MS/"+str(i+1)+".TIF")
#        gt = load_image(data_dir+"Reduced Resolution/GT/"+str(i+1)+".TIF")
        frms = load_image(data_dir+"Full Resolution/MS/"+str(i+1)+".TIF")
        frpan = load_image(data_dir + "Full Resolution/PAN/" + str(i + 1) + ".TIF")
        result = forward(model, pan, ms)
        frresult = forward(model, frpan, frms)
        result = np.round(np.clip(result, 0, 1)*max_value)
        frresult = np.round(np.clip(frresult, 0, 1) * max_value)
        saveimg(img_save_dir+"RR-{}-Fused-{}/".format(method, satellite)+str(i+1)+".TIF", result)
        saveimg(img_save_dir + "FR-{}-Fused-{}/".format(method, satellite) + str(i + 1) + ".TIF", frresult)
