import os.path
from tqdm import tqdm
import glob
import cv2
import torch
import numpy as np
#np.set_printoptions(threshold=np.inf)


root = 'D:/PycharmProjects/pj_orient/data/data0/C0'
#print(os.path.join(root, f'*/patient{1}*'))



data = []



for i in range(37, 41):
    data += glob.glob(root + f'/*/patient{i}*')
print(data)
print(len(data))
img_fname = data[0]
label = int(img_fname.split('\\')[-2])
label = torch.tensor(label).long()
print(label)



# datalist = os.listdir(root)

# img = cv2.imread(os.path.join(root, 'patient1_C00.png'), 0)
# print(img)
# img = torch.tensor(img).float()
# print(img)