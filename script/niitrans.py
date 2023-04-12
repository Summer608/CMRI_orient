"""
将nii文件转化为png
"""
import os
import nibabel as nib
import imageio
import numpy as np

def nii_to_image(niifile):
    filename = os.listdir(niifile)

    for f in filename:
        #读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        fname = f.replace('.nii.gz', '')
        img_f_path = os.path.join(imgfile)

        # 开始转换为图像
        (x, y, z) = img.shape

        for i in range(z):  # z是图像的序列
            silce = img_fdata[:, :, i]
            # imageio.imwrite(os.path.join(img_f_path,'{}.png'.format(i)), silce)
            imageio.imwrite(os.path.join(img_f_path, str(fname)+'{}.png'.format(i)), silce)



if __name__ == '__main__':
    filepath = r'D:\PycharmProjects\pj_orient\data\T2'
    imgfile = r'D:\PycharmProjects\pj_orient\data\data_transform\T2'
    nii_to_image(filepath)