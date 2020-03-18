import argparse
import pandas as pd
from models.get_model import get_arch
from utils.get_loaders import get_test_loader

from utils.evaluation import eval_predictions_multi
from utils.reproducibility import set_seeds
from utils.model_saving_loading import load_model
from tqdm import trange
import numpy as np
import torch
import os.path as osp
import os
import sys


def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--csv_test_q', type=str, default='data/test_q_mt.csv', help='path to test OD data csv')
parser.add_argument('--model_name_MT', type=str, default='resnext50_sws', help='selected architecture')
parser.add_argument('--load_path_MT', type=str, default='experiments/best_auc_MT_qualities_20Mar', help='path to saved model - MT')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='from pretrained weights')
parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=True, help='use tta')
parser.add_argument('--n_classes', type=int, default=5, help='number of target classes (5)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--csv_out', type=str, default='results/submission_quality_galdran_20Mar.csv', help='path to output csv')

args = parser.parse_args()

def run_one_epoch_multi(loader, model, optimizer=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all_quality, preds_all_quality, labels_all_quality = [], [], []
    probs_all_artifact, preds_all_artifact, labels_all_artifact = [], [], []
    probs_all_clarity, preds_all_clarity, labels_all_clarity = [], [], []
    probs_all_field_def, preds_all_field_def, labels_all_field_def = [], [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels_quality, labels_artifact, labels_clarity, labels_field_def) in enumerate(loader):
            inputs, labels_quality, labels_artifact, labels_clarity, labels_field_def  = \
                inputs.to(device, non_blocking=True), labels_quality.to(device, non_blocking=True), \
                labels_artifact.to(device, non_blocking=True), labels_clarity.to(device, non_blocking=True), \
                labels_field_def.to(device, non_blocking=True)

            logits = model(inputs)
            logits_quality = logits[:, :2]
            logits_artifact = logits[:, 2:8]
            logits_clarity = logits[:, 8:13]
            logits_field_def = logits[:, 13:]

            probs_quality = torch.nn.Softmax(dim=1)(logits_quality)
            probs_artifact = torch.nn.Softmax(dim=1)(logits_artifact)
            probs_clarity = torch.nn.Softmax(dim=1)(logits_clarity)
            probs_field_def = torch.nn.Softmax(dim=1)(logits_field_def)

            _, preds_quality = torch.max(probs_quality, 1)
            _, preds_artifact = torch.max(probs_artifact, 1)
            _, preds_clarity = torch.max(probs_clarity, 1)
            _, preds_field_def = torch.max(probs_field_def, 1)

            loss_quality = criterion(logits_quality, labels_quality)
            loss_artifact = criterion(logits_artifact, labels_artifact)
            loss_clarity = criterion(logits_clarity, labels_clarity)
            loss_field_def = criterion(logits_field_def, labels_field_def)

            loss = torch.mean(torch.stack([loss_quality,loss_artifact,loss_clarity,loss_field_def]))

            if train:  # only in training mode
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ll = loss.item()
            del loss
            probs_all_quality.extend(probs_quality.detach().cpu().numpy())
            preds_all_quality.extend(preds_quality.detach().cpu().numpy())
            labels_all_quality.extend(labels_quality.detach().cpu().numpy())

            probs_all_artifact.extend(probs_artifact.detach().cpu().numpy())
            preds_all_artifact.extend(preds_artifact.detach().cpu().numpy())
            labels_all_artifact.extend(labels_artifact.detach().cpu().numpy())

            probs_all_clarity.extend(probs_clarity.detach().cpu().numpy())
            preds_all_clarity.extend(preds_clarity.detach().cpu().numpy())
            labels_all_clarity.extend(labels_clarity.detach().cpu().numpy())

            probs_all_field_def.extend(probs_field_def.detach().cpu().numpy())
            preds_all_field_def.extend(preds_field_def.detach().cpu().numpy())
            labels_all_field_def.extend(labels_field_def.detach().cpu().numpy())


            # Compute running loss
            running_loss += ll * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(tr_loss="{:.4f}".format(float(run_loss)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    return np.stack(preds_all_quality), np.stack(probs_all_quality), np.stack(labels_all_quality), \
           np.stack(preds_all_artifact), np.stack(probs_all_artifact), np.stack(labels_all_artifact), \
           np.stack(preds_all_clarity), np.stack(probs_all_clarity), np.stack(labels_all_clarity), \
           np.stack(preds_all_field_def), np.stack(probs_all_field_def), np.stack(labels_all_field_def), run_loss

def run_one_epoch_cls(loader, model, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    probs_all, preds_all, labels_all = [], [], []
    with trange(len(loader)) as t:
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            _, preds = torch.max(probs, 1)
            probs_all.extend(probs.detach().cpu().numpy())
            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(labels.detach().cpu().numpy())
            run_loss = 0
            t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    return np.stack(preds_all), np.stack(probs_all), np.stack(labels_all)

def test_cls_tta_dihedral_MT(model, test_loader, n=3):
    probs_tta_q, probs_tta_a, probs_tta_c, probs_tta_f = [], [], [], []
    prs = [0, 1]
    import torchvision
    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                # test_preds, test_probs, test_labels = run_one_epoch_multi(test_loader, model)
                _, probs_q, _, \
                _, probs_a, _, \
                _, probs_c, _, \
                _, probs_f, _, _ = run_one_epoch_multi(test_loader, model)
                probs_tta_q.append(probs_q)
                probs_tta_a.append(probs_a)
                probs_tta_c.append(probs_c)
                probs_tta_f.append(probs_f)

    probs_tta_q = np.mean(np.array(probs_tta_q), axis=0)
    preds_tta_q = np.argmax(probs_tta_q, axis=1)

    probs_tta_a = np.mean(np.array(probs_tta_a), axis=0)
    preds_tta_a = np.argmax(probs_tta_a, axis=1)

    probs_tta_c = np.mean(np.array(probs_tta_c), axis=0)
    preds_tta_c = np.argmax(probs_tta_c, axis=1)

    probs_tta_f = np.mean(np.array(probs_tta_f), axis=0)
    preds_tta_f = np.argmax(probs_tta_f, axis=1)

    del model
    torch.cuda.empty_cache()
    return probs_tta_q, preds_tta_q, probs_tta_a, preds_tta_a, probs_tta_c, preds_tta_c, probs_tta_f, preds_tta_f

def test_cls_tta_dihedral(model, test_loader, n=3):
    probs_tta = []
    prs = [0, 1]
    import torchvision
    test_loader.dataset.transforms.transforms.insert(-1, torchvision.transforms.RandomRotation(0))
    rotations = np.array([i * 360 // n for i in range(n)])
    for angle in rotations:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            test_loader.dataset.transforms.transforms[-2].degrees = [angle, angle]
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs)

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = np.argmax(probs_tta, axis=1)
    # test_k, test_auc, test_acc = eval_predictions_multi(test_labels, preds_tta, probs_tta)
    # print('Test Kappa: {:.4f} -- AUC: {:.4f} -- Balanced Acc: {:.4f}'.format(test_k, test_auc, test_acc))

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        preds, probs, labels = run_one_epoch_cls(test_loader, model)
    # vl_k, vl_auc, vl_acc = eval_predictions_multi(labels, preds, probs)
    # print('Val. Kappa: {:.4f} -- AUC: {:.4f}'.format(vl_k, vl_auc).rstrip('0'))

    del model
    torch.cuda.empty_cache()
    return probs, preds, labels

if __name__ == '__main__':
    '''
    Example:
    python test.py --tta True --csv_out results/submission_galdran_11Mar.csv
    '''
    data_path = 'data'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name_MT = args.model_name_MT
    load_path_MT = args.load_path_MT
    pretrained = args.pretrained
    bs = args.batch_size
    csv_test_q = args.csv_test_q
    n_classes = args.n_classes
    tta = args.tta
    csv_out = args.csv_out

    ####################################################################################################################
    # build results for MT model
    n_classes = 18
    print('* Instantiating MT model {}, pretrained={}'.format(model_name_MT, pretrained))
    model, mean, std = get_arch(model_name_MT, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_MT, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_q,  batch_size=bs, mean=mean, std=std, qualities=True)

    probs_tta_q, preds_tta_q, probs_tta_a, preds_tta_a, probs_tta_c, preds_tta_c, probs_tta_f, preds_tta_f \
        = test_cls_tta_dihedral_MT(model, test_loader, n=3)

    df_quality = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_tta_q), columns=['image_id', 'Overall quality'])

    df_artifact = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_tta_a), columns=['image_id', 'Artifact'])
    def map_label_art(label):
        if label == 0:  return 0
        elif label == 1: return 1
        elif label == 2: return 4
        elif label == 3: return 6
        elif label == 4: return 8
        else: return 10

    df_artifact['Artifact'] = df_artifact['Artifact'].apply(map_label_art)
    df_clarity = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_tta_c), columns=['image_id', 'Clarity'])
    def map_label_clarity(lab):
        if lab == 0: return 1
        elif lab == 1: return 4
        elif lab == 2: return 6
        elif lab == 3: return 8
        else: return 10
    df_clarity['Clarity'] = df_clarity['Clarity'].apply(map_label_clarity)

    df_field_def = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_tta_f), columns=['image_id', 'Field definition'])
    def map_label_fd(label):
        if label == 0: return 1
        elif label == 1: return 4
        elif label == 2: return 6
        elif label == 3: return 8
        else: return 10
    df_field_def['Field definition'] = df_field_def['Field definition'].apply(map_label_fd)

    from functools import reduce

    submission = reduce(lambda x, y: pd.merge(x, y, on='image_id'), [df_quality, df_artifact, df_clarity, df_field_def])
    submission['image_id'] = submission['image_id'].apply(lambda x: x.split('/')[-1][:-4])
    submission = submission[['Overall quality', 'Artifact', 'Clarity', 'Field definition', 'image_id']]
    submission.to_csv(csv_out, index=False)