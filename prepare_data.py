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
    path_raw_data = '/home/agaldran/Desktop/data/deepdr_data/'
    path_data_out = 'data/'

    path_out_ims = osp.join(path_data_out, 'images')
    path_out_fovs = osp.join(path_data_out, 'fovs')
    os.makedirs(path_out_ims, exist_ok=True)
    os.makedirs(path_out_fovs, exist_ok=True)

    print('Preparing training data')
    csv_path = osp.join(path_raw_data, 'regular-fundus-training/regular-fundus-training.csv')

    df = pd.read_csv(csv_path)
    df = df.drop(['patient_id', 'Overall quality', 'patient_DR_Level',
                  'Clarity', 'Field definition', 'Artifact'], axis=1)
    df.left_eye_DR_Level.fillna(0, inplace=True)
    df.right_eye_DR_Level.fillna(0, inplace=True)
    df['dr'] = df['left_eye_DR_Level'] + df['right_eye_DR_Level']
    df = df.drop(['left_eye_DR_Level', 'right_eye_DR_Level'], axis=1)
    df['center'] = ['od' if '1.jpg' in n else 'mac' for n in list(df.image_path)]
    abs_path = [osp.join(path_raw_data, n.replace('\\', '/')[1:]) for n in list(df.image_path)]
    df['abs_path'] = abs_path
    df = df.drop(['image_path'], axis=1)
    df_od = df[df.center == 'od']
    df_mac = df[df.center == 'mac']
    im_list = df.abs_path

    num_ims = len(im_list)
    # Parallel(n_jobs=6)(delayed(process_im_fast)(i, im_list, path_out_ims, path_out_fovs)
    #                    for i in tqdm(range(num_ims)))
    image_ids = [osp.join(path_out_ims, im_name.split('/')[-1]) for im_name in im_list]
    df['image_id'] = image_ids
    df = df.drop(['abs_path'], axis=1)

    df['dr'] = df['dr'].astype(int)

    df['center'] = ['od' if '1.jpg' in n else 'mac' for n in list(df.image_id)]
    df_od = df[df.center == 'od']
    df_mac = df[df.center == 'mac']
    df_od.drop(['center'], axis=1).to_csv(osp.join(path_data_out,'train_od.csv'), index=False)
    df_mac.drop(['center'], axis=1).to_csv(osp.join(path_data_out, 'train_mac.csv'), index=False)
    print('Training data prepared')

    print('Preparing validation data')
    csv_path = osp.join(path_raw_data, 'regular-fundus-validation/regular-fundus-validation.csv')

    df = pd.read_csv(csv_path)
    df = df.drop(['patient_id', 'Overall quality', 'patient_DR_Level',
                  'Field definition.1', 'Field definition', 'Artifact'], axis=1)
    df.left_eye_DR_Level.fillna(0, inplace=True)
    df.right_eye_DR_Level.fillna(0, inplace=True)
    df['dr'] = df['left_eye_DR_Level'] + df['right_eye_DR_Level']
    df = df.drop(['left_eye_DR_Level', 'right_eye_DR_Level'], axis=1)
    df['center'] = ['od' if '1.jpg' in n else 'mac' for n in list(df.image_path)]
    abs_path = [osp.join(path_raw_data, n.replace('\\', '/')[1:]) for n in list(df.image_path)]
    df['abs_path'] = abs_path
    df = df.drop(['image_path'], axis=1)
    df_od = df[df.center == 'od']
    df_mac = df[df.center == 'mac']
    im_list = df.abs_path

    num_ims = len(im_list)
    Parallel(n_jobs=6)(delayed(process_im_fast)(i, im_list, path_out_ims, path_out_fovs)
                       for i in tqdm(range(num_ims)))
    image_ids = [osp.join(path_out_ims, im_name.split('/')[-1]) for im_name in im_list]
    df['image_id'] = image_ids
    df = df.drop(['abs_path'], axis=1)

    df['dr'] = df['dr'].astype(int)

    df['center'] = ['od' if '1.jpg' in n else 'mac' for n in list(df.image_id)]
    df_od = df[df.center == 'od']
    df_mac = df[df.center == 'mac']
    df_od.drop(['center'], axis=1).to_csv(osp.join(path_data_out, 'val_od.csv'), index=False)
    df_mac.drop(['center'], axis=1).to_csv(osp.join(path_data_out, 'val_mac.csv'), index=False)
    print('Validation data prepared')



