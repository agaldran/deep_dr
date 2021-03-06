import pandas as pd
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import warnings
from skimage.measure import regionprops, label
from skimage import io
from skimage.filters import threshold_li
from scipy.ndimage import binary_fill_holes
from skimage.exposure import adjust_gamma as gamma
from PIL import Image
from torchvision import transforms as tr
from joblib import Parallel, delayed


tg_size = (512, 512)
rsz = tr.Resize(tg_size)
rsz_fov = tr.Resize(tg_size, interpolation=Image.NEAREST)


def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

from skimage import draw
from scipy import optimize
from skimage import img_as_float
from skimage.filters import threshold_minimum

def get_circle(binary):
    regions = regionprops(binary.astype(int), coordinates='rc')
    bubble = regions[0]

    x0, y0 = bubble.centroid
    r = bubble.major_axis_length / 2.

    def cost(params):
        x0, y0, r = params
        coords = draw.circle(y0, x0, r, shape=binary.shape)
        template = np.zeros_like(binary)
        template[coords] = 1
        return -np.sum(template == binary)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r), disp=False)

    return x0, y0, r

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def process_im_fast(i, im_list, path_out_ims, path_out_fovs):
    im_name = im_list[i]
    im_name_out = osp.join(path_out_ims, im_name.split('/')[-1])
    im = Image.open(im_name)
    rsz_512 = tr.Resize(512)
    im_rsz = rsz_512(im)
    ctr_crop = tr.CenterCrop([512, 512])
    ctr_crop(im_rsz).save(im_name_out)

def process_im(i, im_list, path_out_ims, path_out_fovs):
    im_name = im_list[i]
    im_name_out = osp.join(path_out_ims, im_name.split('/')[-1])
    fov_name_out = osp.join(path_out_fovs, im_name.split('/')[-1][:-3] + 'png')
    im = Image.open(im_name)
    h, w = im.size
    rsz_1000 = tr.Resize([1000, 1000])

    im_red = img_as_float(np.array(rsz_1000(im)))
    im_v = im_red[:, :, 1] ** 0.5
    thresh = threshold_li(im_v)

    binary = im_v > thresh
    x0, y0, r = get_circle(binary)
    binary = create_circular_mask(binary.shape[0], binary.shape[1], (x0, y0), int(1.025*r))

    rsz_back = tr.Resize([w, h], interpolation=Image.NEAREST)
    binary = np.array(rsz_back(Image.fromarray(binary)))

    label_img = label(binary)
    regions = regionprops(label_img)
    areas = [r.area for r in regions]
    largest_cc_idx = np.argmax(areas)

    fov = regions[largest_cc_idx]

    cropped = np.array(im)[fov.bbox[0]:fov.bbox[2], fov.bbox[1]: fov.bbox[3], :]

    rsz(Image.fromarray(cropped)).save(im_name_out)
    rsz_fov(Image.fromarray(fov.filled_image)).save(fov_name_out)

if __name__ == "__main__":
    # handle paths
    path_raw_data = '/home/agaldran/Desktop/ISBI2020_raw_data/Onsite-Challenge1&2-Evaluation'
    path_data_out = 'data/'

    path_out_ims = osp.join(path_data_out, 'test_images_ONSITE')
    path_out_fovs = osp.join(path_data_out, 'test_fovs_ONSITE')
    os.makedirs(path_out_ims, exist_ok=True)
    # os.makedirs(path_out_fovs, exist_ok=True)

    print('Preparing test data')
    csv_path = osp.join(path_raw_data, 'Challenge1_upload.csv')
    df = pd.read_csv(csv_path)
    image_list = df.image_id.values
    image_path_list = df.image_id.values
    image_path_list = [n[:3] for n in image_path_list]

    im_list = [osp.join(path_raw_data, n1, n2+'.jpg') for n1, n2 in zip(image_path_list, image_list)]
    num_ims = len(im_list)
    # Parallel(n_jobs=6)(delayed(process_im_fast)(i, im_list, path_out_ims, path_out_fovs)
    #                    for i in tqdm(range(num_ims)))

    fake_grades = len(df['image_id'])*[0]
    # fake_grades[1], fake_grades[2] = 1, 1
    # fake_grades[3], fake_grades[4] = 2, 2
    # fake_grades[5], fake_grades[6] = 3, 3
    # fake_grades[7], fake_grades[8] = 4, 4
    df['dr'] = fake_grades


    df['center'] = ['od' if 'l1' in n or 'r1' in n else 'mac' for n in list(df.image_id)]

    df_od = df[df.center == 'od']
    df_mac = df[df.center == 'mac']

    pd.options.mode.chained_assignment = None
    df_od['image_id'] = df_od['image_id'].apply(lambda x: 'data/test_images_ONSITE/' + x + '.jpg')
    df_mac['image_id'] = df_mac['image_id'].apply(lambda x: 'data/test_images_ONSITE/' + x + '.jpg')

    df_od.drop(['center', 'DR_Level'], axis=1).to_csv(osp.join(path_data_out, 'test_od_ONSITEE.csv'), index=False)
    df_mac.drop(['center', 'DR_Level'], axis=1).to_csv(osp.join(path_data_out, 'test_mac_ONSITEE.csv'), index=False)
    print('Test data prepared')

    csv_path = osp.join(path_raw_data, 'Challenge2_upload.csv')
    df_q = pd.read_csv(csv_path)
    df_q['image_id'] = df_q['image_id'].apply(lambda x: 'data/test_images_ONSITE/' + x + '.jpg')
    fake_grades = len(df_q['image_id']) * [0]
    df_q['dr'] = fake_grades
    df_q = df_q.drop(['Overall quality','Artifact','Clarity','Field definition'], axis=1)
    df_q.to_csv('data/test_q_ONSITEE.csv', index=False)
    df_q['quality'] = df_q['dr']
    df_q['artifact'] = df_q['dr']
    df_q['clarity'] = df_q['dr']
    df_q['field_def'] = df_q['dr']
    df_q = df_q.drop(['dr'], axis=1)
    df_q.to_csv('data/test_q_mt_ONSITEE.csv', index=False)
    print('Test data for quality assessment prepared')