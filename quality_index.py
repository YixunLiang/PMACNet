import metrics
from osgeo import gdal
import os
import tqdm
import numpy as np
from config import TrainGlobalConfig
config = TrainGlobalConfig

satellite = config.satellite
method = config.method
FR_test_dir = 'datasets/{}/test/Full Resolution/'.format(config.satellite)
RR_test_dir = 'datasets/{}/test/Reduced Resolution/'.format(config.satellite)
FR_fused_dir = 'gen_imgs/-{}-{}/FR-{}-Fused-{}/'.format(config.method,config.satellite,config.method,config.satellite)
RR_fused_dir = 'gen_imgs/-{}-{}/RR-{}-Fused-{}/'.format(config.method,config.satellite,config.method,config.satellite)


def load_image(filepath):
    dataset = gdal.Open(filepath)
    img = dataset.ReadAsArray()
    return img


length =100
scc = np.zeros((length,))
#GT = load_image(FR_test_dir+'GT/'+str(1)+'.TIF')
#print(GT.shape)
sam = np.zeros((length,))

ergas = np.zeros((length,))
q8 = np.zeros((length,))
for i in tqdm.tqdm(range(length)):
    GT = load_image(RR_test_dir+'GT/'+str(i+1)+'.TIF')
    PAN = load_image(RR_test_dir + 'PAN/'+str(i+1)+'.TIF')
    MS = load_image(RR_test_dir + 'MS/'+str(i+1)+'.TIF')
    Fused = load_image(RR_fused_dir + str(i + 1) + '.TIF')
    scc[i] = metrics.SCC(GT, Fused)
    sam[i] = metrics.SAM(GT, Fused)
    ergas[i] = metrics.ERGAS(GT, Fused)
    q8[i] = metrics.Q2n(GT, Fused)
#    print("SCC:{:.4f} SAM:{:.4f} ERGAS:{:.4f} Q8:{:.4f}".format(scc[i], sam[i], ergas[i], q8[i]))
#    os.system("pause")
print("SCC:{:.4f} SAM:{:.4f} ERGAS:{:.4f} Q8:{:.4f}".format(np.mean(scc), np.mean(sam), np.mean(ergas), np.mean(q8)))

D_s = np.zeros((length,))
D_lambda = np.zeros((length,))
QNR = np.zeros((length,))
for i in tqdm.tqdm(range(length)):
    PAN = load_image(FR_test_dir + 'PAN/'+str(i+1)+'.TIF')
    MS = load_image(FR_test_dir + 'MS/'+str(i+1)+'.TIF')
    Fused = load_image(FR_fused_dir + str(i+1) + '.TIF')
    QNR[i], D_s[i], D_lambda[i] = metrics.QNR(Fused, MS, PAN)
#    print("QNR:{:.4f} D_lambda:{:.4f} D_s:{:.4f}".format(QNR[i], D_lambda[i], D_s[i]))
#    os.system("pause")
print("QNR:{:.4f} D_lambda:{:.4f} D_s:{:.4f}".format(np.mean(QNR), np.mean(D_lambda), np.mean(D_s)))