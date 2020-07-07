
import os
import json
import numpy as np
from mate.load_input_data import load_input_data
from mate.preprocess_lung_part import preprocess_lung_part
from lib.threshold_function_module import windowlize_image
from lib.png_rw import npy_to_png
from lib.judge_mkdir import judge_mkdir
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shutil import copy2



def move_and_process_dcm_data(source,output='/home/yujwu/Data/NLST/survivalestimate/data/output_nodule',destination='./dcm_data'):
    files = os.listdir(source)
    for i in files:
        copy2(f"{source}/{i}", destination)
        image,image_raw,info_dict,img = Load_preprocess_raw_data(destination)
        lesion_np_path, lung_np_path = process_lung_part(info_dict,image_raw)
        input_dict = {"image_np_path": [[lesion_np_path, lung_np_path]]}
        segmentation(input_dict,img,i,output)
        os.remove(f"./dcm_data/{i}")


def Load_preprocess_raw_data(filepath='./dcm_data'):
    # image_raw, info_dict = load_input_data('./dcm_data')
    image_raw, info_dict = load_input_data(filepath)
    # 1. 展示医学图像
    image = windowlize_image(image_raw, 1500, -500)[0]
    image = npy_to_png(image)
    image = (image - float(np.min(image))) / float(np.max(image)) * 255.

    image = image[np.newaxis, :, :]
    image = image.transpose((1, 2, 0)).astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.imwrite('demo.png', image)
    img = mpimg.imread('demo.png')

    return image,image_raw,info_dict,img


def process_lung_part(info_dict,image_raw):
    # 2. 进行肺部分割需要的前处理
    lung_part, info_dict = preprocess_lung_part(image_raw, info_dict)
    # 3. 进行病灶分割需要的前处理
    ww, wc = (1500, -500)
    lesion_part = windowlize_image(image_raw.copy(), ww, wc)
    lesion_part = np.squeeze(lesion_part, 0)
    lesion_np_path = "lesion_part.npy"
    lung_np_path = "lung_part.npy"
    np.save(lung_np_path, lung_part)
    np.save(lesion_np_path, lesion_part)

    return lesion_np_path, lung_np_path

def segmentation(input_dict,img,i,output):
    import paddlehub as hub
    pneumonia = hub.Module(name="Pneumonia_CT_LKM_PP")
    results = pneumonia.segmentation(data=input_dict)

    lesion_part = results[0]['output_lesion_np'].astype(np.uint8)
    lesion_mask=results[0]['output_lesion_np'].astype(np.uint8)
    mask_row = 30
    mask_col = 30
    #create a new image for storing the image data in the mask region
    image_new = np.zeros([512,512,3]).astype(np.float32)
    # get the index of mask non-zero
    indx = np.nonzero(lesion_mask)
    xindex = len(indx[0])
    yindex = len(indx[1])

    if xindex != 0:
        xindex_min = min(indx[0])
        xindex_max = max(indx[0])

        yindex_min = min(indx[1])
        yindex_max = max(indx[1])

        index_x = [i for i in range(xindex_min,xindex_max)]
        index_y = [i for i in range(yindex_min,yindex_max)]

        x_i = []
        y_i = []

        for x in index_x:
            for y in index_y:
                y_i.append(y)
                x_i.append(x)

        img2 = img[x_i, y_i, :]

        size_test = len(indx[0])
        # img2=np.reshape(img2,[int(size_test/1),1,3])
        img2 = np.reshape(img2, [len(index_x), len(index_y), 3])
        plt.imshow(img2)
        plt.title('This is the leision seperate0', color='blue')
        plt.show()

        # img3 = npy_to_png(img2)

        plt.imsave(f'{output}/{i[:-3]}png',img2)
        # img_t = mpimg.imread(f'{output}/{i[:-3]}png')
        # plt.imshow(img_t)
        # plt.title('This is the leision seperate1', color='blue')
        # plt.show()
    else:
        print("no leision")


move_and_process_dcm_data("/home/yujwu/Data/NLST/preprocessing/Pneumonia-CT-LKM-PP/aistudio/more_data")

exit(0)
# # image_new[indx[0],indx[1],:] = img[index_x,index_y,:]
# image_new[indx[0],indx[1],:] = img[indx[0],indx[1],:]
#
# # # create  a new image that have the same dimension of mask non-zero region
# # # img2 = np.zeros([np.max(indx[0])-np.min(indx[0])+1,np.max(indx[1])-np.min(indx[1])+1,3])
# # # save the mask region into img2
# # img2= image_new[indx[0],indx[1],:]
# img2 = img[x_i,y_i,:]
#
# size_test = len(indx[0])
# # img2=np.reshape(img2,[int(size_test/1),1,3])
# img2=np.reshape(img2,[len(index_x),len(index_y),3])
#
# plt.imshow(img2)
# plt.title('This is the leision kouchulaide',color='blue')
# plt.show()





from PIL import Image as PILImage
from mate.postprocess_lung_part import postprocess_lung_part
from mate.merge_process import merge_process
from lib.remove_small_obj_module import remove_small_obj


# 将类别转换为可视化的像素点值
def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map


color_map_lesion = get_color_map_list(num_classes=2)
color_map_lung = get_color_map_list(num_classes=3)

lung_part = postprocess_lung_part(results[0]['output_lung_np'], info_dict)

lesion_part = results[0]['output_lesion_np'].astype(np.uint8)
# for i in range(len(lesion_part)):
#     lesion_part[i] = remove_small_obj(lesion_part[i], 10)

plt.title('This is the leision part',color='blue')
plt.imshow(lesion_part)
plt.axis('off')
plt.show()

# 对肺部分割结果和病灶分割结果进行后处理
lung_part, lesion_part = merge_process(image_raw, lung_part, lesion_part)


test = np.argmax(lung_part, -1)[0].astype(np.uint8)
lesion_part_test = lesion_part[0].astype(np.uint8)

pred_mask = PILImage.fromarray(lesion_part_test, mode='P')#lung_part
pred_mask.putpalette(color_map_lung)#color_map_lesion
pred_mask = pred_mask.convert('RGB')


lung_merge_img = np.where(pred_mask, pred_mask, img)
fig, axarr = plt.subplots(1, 1, figsize=(10, 10))


##axarr.imshow(img)
plt.imshow(lung_merge_img)
plt.axis('off')
plt.show()


axarr.axis('off')
axarr.imshow(lung_merge_img)

exit(0)

resmap = results[0]['output_lesion_np']
pred_mask = PILImage.fromarray(resmap.astype(np.uint8), mode='P')
pred_mask.putpalette(color_map_lesion)

pred_mask = pred_mask.convert('RGB')

plt.title('This is the leision mask',color='blue')
plt.imshow(pred_mask)
plt.axis('off')
plt.show()

lesion_merge_img = np.where(pred_mask, pred_mask, img)

fig, axarr = plt.subplots(1, 1, figsize=(10, 10))

axarr.axis('off')
axarr.imshow(lesion_merge_img)

import json
import numpy as np
from lib.info_dict_module import InfoDict
from mate.save_merged_png_cv2 import save_merged_png_cv2
from lib.judge_mkdir import judge_mkdir

# 融合肺部分割结果和病灶分割结果

image = windowlize_image(image_raw, 1500, -500)[0]
image = npy_to_png(image)
image = (image - float(np.min(image))) / float(np.max(image)) * 255.

lung = lung_part[0,..., 1] + lung_part[0,..., 2]
binary = lung * 255
binary = binary.astype(np.uint8)
try:
    _, lung_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
except:
    lung_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

binary = lesion_part[0] * 255
binary = binary.astype(np.uint8)

try:
    _, lesion_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
except:
    lesion_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image = image[np.newaxis, :, :]
image = image.transpose((1, 2, 0)).astype('float32')
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

cv2.drawContours(image, lesion_contours, -1, (0, 0, 255), 2)
cv2.drawContours(image, lung_contours, -1, (0, 255, 0), 2)

cv2.imwrite('merged.png', image)
# 可以看到此时误检的非肺部分已经被去除
img = mpimg.imread('merged.png')
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()

# 最后我们根据预测计算一下病灶占比，病灶体积，病灶个数
from lib.c1_cal_lesion_percent import cal_lesion_percent
from lib.c2_cal_lesion_volume import cal_lesion_volume
from lib.c3_cal_lesion_num import cal_lesion_num
from lib.c4_cal_histogram import cal_histogram
from lib.c5_normal_statistics import normal_statistics


def cal_metrics(image_raw, lung_part, lesion_part, spacing_list):
    """
    进行指标计算
    整体流程：
    1. 分别得到左右肺和左右病灶
    2. 计算病灶占比
    3. 计算病灶体积
    4. 计算病灶个数
    5. 计算直方图
    """
    print('cal the statistics metrics')
    # 1. 分别得到左右肺和左右病灶
    lung_l = lung_part[..., 1]
    lung_r = lung_part[..., 2]
    lesion_l = lesion_part.copy() * lung_l
    lesion_r = lesion_part.copy() * lung_r

    lung_tuple = (lung_l, lung_r, lung_part)
    lesion_tuple = (lesion_l, lesion_r, lesion_part)

    # t = image_temp
    # t1 = lung_part[0]
    #
    # cv2.imwrite('demo.png', lung_part[0])
    # img = mpimg.imread('demo.png')
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title('This is lung_part', color='blue')
    # plt.show()

    # img = cv2.imread(image_temp)
    # img = image_temp
    # cv2.imwrite('demo.png', img)
    # img = cv2.imread('demo.png')
    # # rect = (0, 0, 300, 300)
    # rect = (275, 120, 170, 320)
    # mask9 = np.zeros(img.shape[:2], np.uint8)
    # mask = lesion_part[0]
    # bgModel = np.zeros((1, 65), np.float64)
    # fgModel = np.zeros((1, 65), np.float64)
    # cv2.grabCut(img, np.float32(mask), None, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    #
    # out = img * mask2[:, :, np.newaxis]
    #
    # cv2.imshow('output', out)
    # cv2.waitKey()


    # 2. 计算病灶占比
    lesion_percent_dict = cal_lesion_percent(lung_tuple, lesion_tuple)

    # 3. 计算病灶体积
    lesion_volume_dict = cal_lesion_volume(lesion_tuple, spacing_list)

    # 4. 计算病灶个数
    lesion_num_dict = cal_lesion_num(lesion_tuple)

    # 5. 计算直方图
    hu_statistics_dict = cal_histogram(image_raw, lung_tuple)

    metrics_dict = {
        'lesion_num': lesion_num_dict,
        'lesion_volume': lesion_volume_dict,
        'lesion_percent': lesion_percent_dict,
        'hu_statistics': hu_statistics_dict,
        'normal_statistics': normal_statistics
    }

    return metrics_dict


# 进行指标计算
metrics_dict = cal_metrics(image_raw, lung_part, lesion_part, info_dict.spacing_list)
# 打印一下各项指标, 'lung_l'为左肺，'lung_r'为右肺， 'lung_all'为两个肺。
print('病灶个数', metrics_dict['lesion_num'])
print('病灶体积', metrics_dict['lesion_volume'])
print('病灶占比', metrics_dict['lesion_percent'])
print('lesion_part',lesion_part)

# exit(0)
# fname = 'images/test1.jpg'
# img = cv2.imread(image_raw)
# rect = lesion_part
#
# mask = np.zeros(img.shape[:2], np.uint8)
# bgModel = np.zeros((1,65), np.float64)
# fgModel = np.zeros((1,65), np.float64)
# cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
#
# out = img * mask2[:, :, np.newaxis]
#
# cv2.imshow('output', out)
# cv2.waitKey()

