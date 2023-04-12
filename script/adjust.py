import numpy as np
import cv2
import os
from tqdm import tqdm


def adjust(file: str, direct: int):
    savepath = os.path.join(target, str(direct))
    makedir(savepath)
    img = cv2.imread(os.path.join(root, file), 0) #读取灰度图

    if direct == 0:
        return
    elif direct == 1:
        new = np.fliplr(img)
    elif direct == 2:
        new = np.flipud(img)  # 010 Target[x,y,z]=Source[x,sy-y,z]
    elif direct == 3:
        new = np.flipud(np.fliplr(img))  # 011 Target[x,y,z]=Source[sx-x,sy-y,z]
    elif direct == 4:
        new = img.T  # 100 Target[x,y,z]=Source[y,x,z]
    elif direct == 5:
        # 101 Target[x,y,z]=Source[sx-y,x,z]
        new = np.fliplr(img.T)
    elif direct == 6:
        # 110 Target[x,y,z]=Source[y,sy-x,z]
        new = np.flipud(img.T)
    elif direct == 7:
        new = np.flipud(np.fliplr(img.T))  # 111 Target[x,y,z]=Source[sx-y,sy-x,z]
    cv2.imwrite(os.path.join(savepath, file), new)

def makedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == "__main__":
    root = r'D:\PycharmProjects\pj_orient\data\data_transform\T2'
    target = r'D:\PycharmProjects\pj_orient\data\data0\T2'

    datalist = os.listdir(root)
    for file in tqdm(datalist):
        for i in range(1, 8):
            adjust(file, direct=i)
