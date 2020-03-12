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
    df_od = df[df.center == 'od'].drop(['center'], axis=1)
    df_mac = df[df.center == 'mac'].drop(['center'], axis=1)
    df_od.to_csv(osp.join(path_data_out, 'train_od.csv'), index=False)
    df_mac.to_csv(osp.join(path_data_out, 'train_mac.csv'), index=False)
    pd.concat([df_od, df_mac], axis=0).to_csv(osp.join(path_data_out, 'train_all.csv'), index=False)
    print('Training data prepared')

    print('Preparing validation data')
    csv_path = osp.join(path_raw_data, 'regular-fundus-validation/regular-fundus-validation.csv')

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
    df_od = df[df.center == 'od'].drop(['center'], axis=1)
    df_mac = df[df.center == 'mac'].drop(['center'], axis=1)
    df_od.to_csv(osp.join(path_data_out, 'val_od.csv'), index=False)
    df_mac.to_csv(osp.join(path_data_out, 'val_mac.csv'), index=False)
    pd.concat([df_od, df_mac], axis=0).to_csv(osp.join(path_data_out, 'val_all.csv'), index=False)
    print('Validation data prepared')

    print('Preparing image quality data')
    path_raw_data = '/home/agaldran/Desktop/data/deepdr_data/'
    path_data_out = 'data/'

    print('------------------------------------')
    print('Preparing training data')
    pd.options.mode.chained_assignment = None
    csv_path = osp.join(path_raw_data, 'regular-fundus-training/regular-fundus-training.csv')
    df = pd.read_csv(csv_path)
    df = df.drop(['patient_id', 'left_eye_DR_Level', 'right_eye_DR_Level', 'patient_DR_Level'], axis=1)
    abs_path = [osp.join(path_raw_data, n.replace('\\', '/')[1:]) for n in list(df.image_path)]
    df['abs_path'] = abs_path
    df = df.drop(['image_path'], axis=1)
    im_list = df.abs_path
    image_ids = [osp.join(path_out_ims, im_name.split('/')[-1]) for im_name in im_list]
    df['image_id'] = image_ids
    df = df.drop(['abs_path'], axis=1)

    df_qual = df[['image_id', 'Overall quality']]
    df_qual.columns = ['image_id', 'dr']
    df_qual.to_csv(osp.join(path_data_out, 'train_quality.csv'), index=False)

    def map_label_clarity(lab):
        if lab == 1: return 0
        elif lab == 4: return 1
        elif lab == 6: return 2
        elif lab == 8: return 3
        else: return 4
    df_clarity = df[['image_id', 'Clarity']]
    df_clarity['dr'] = df_clarity['Clarity'].apply(map_label_clarity)
    df_clarity = df_clarity.drop(['Clarity'], axis=1)
    df_clarity.to_csv(osp.join(path_data_out, 'train_clarity.csv'), index=False)

    def map_label_fd(label):
        if label == 1: return 0
        elif label == 4: return 1
        elif label == 6: return 2
        elif label == 8: return 3
        else: return 4
    df_field_def = df[['image_id', 'Field definition']]
    df_field_def['dr'] = df_field_def['Field definition'].apply(map_label_fd)
    df_field_def = df_field_def.drop(['Field definition'], axis=1)
    ########### WARNING: DO NOT REWRITE THESE, MANUALLY MODIFIED TO MOVE A 0-CLASS EX TO THE VAL SET
    # df_field_def.to_csv(osp.join(path_data_out,'train_field_def.csv'), index=False)

    def map_label_art(label):
        if label == 0: return 0
        elif label == 1: return 1
        elif label == 4: return 2
        elif label == 6: return 3
        elif label == 8: return 4
        else: return 5
    df_artifact = df[['image_id', 'Artifact']]
    df_artifact['dr'] = df_artifact['Artifact'].apply(map_label_art)
    df_artifact = df_artifact.drop(['Artifact'], axis=1)
    df_artifact.to_csv(osp.join(path_data_out, 'train_artifact.csv'), index=False)


    print('Preparing validation data')
    csv_path = osp.join(path_raw_data, 'regular-fundus-validation/regular-fundus-validation.csv')
    df = pd.read_csv(csv_path)
    df = df.drop(['patient_id', 'left_eye_DR_Level', 'right_eye_DR_Level', 'patient_DR_Level'], axis=1)
    abs_path = [osp.join(path_raw_data, n.replace('\\', '/')[1:]) for n in list(df.image_path)]
    df['abs_path'] = abs_path
    df = df.drop(['image_path'], axis=1)
    im_list = df.abs_path
    image_ids = [osp.join(path_out_ims, im_name.split('/')[-1]) for im_name in im_list]
    df['image_id'] = image_ids
    df = df.drop(['abs_path'], axis=1)

    df_qual = df[['image_id', 'Overall quality']]
    df_qual.columns = ['image_id', 'dr']
    df_qual.to_csv(osp.join(path_data_out,'val_quality.csv'), index=False)

    df_clarity = df[['image_id', 'Clarity']]
    df_clarity['dr'] = df_clarity['Clarity'].apply(map_label_clarity)
    df_clarity = df_clarity.drop(['Clarity'], axis=1)
    df_clarity.to_csv(osp.join(path_data_out, 'val_clarity.csv'), index=False)

    df_field_def = df[['image_id', 'Field definition']]
    df_field_def['dr'] = df_field_def['Field definition'].apply(map_label_fd)
    df_field_def = df_field_def.drop(['Field definition'], axis=1)
    ########### WARNING: DO NOT REWRITE THESE, MANUALLY MODIFIED TO MOVE A 0-CLASS EX TO THE VAL SET
    # df_field_def.to_csv('data/val_field_def.csv', index=False)

    df_artifact = df[['image_id', 'Artifact']]
    df_artifact['dr'] = df_artifact['Artifact'].apply(map_label_art)
    df_artifact = df_artifact.drop(['Artifact'], axis=1)
    df_artifact.to_csv(osp.join(path_data_out, 'val_artifact.csv'), index=False)

    print('Done')