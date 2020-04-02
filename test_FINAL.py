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
parser.add_argument('--csv_test_od', type=str, default='data/test_od.csv', help='path to test OD data csv')
parser.add_argument('--csv_test_mac', type=str, default='data/test_mac.csv', help='path to test MAC data csv')
parser.add_argument('--model_name', type=str, default='resnet50', help='selected architecture')
parser.add_argument('--load_path_od_k', type=str, default='experiments/best_kappa_od_20Mar', help='path to saved model - od1')
parser.add_argument('--load_path_od_auc', type=str, default='experiments/best_auc_od_20Mar', help='path to saved model - od2')
parser.add_argument('--load_path_mac_k', type=str, default='experiments/best_kappa_mac_20Mar', help='path to saved model - mac1')
parser.add_argument('--load_path_mac_auc', type=str, default='experiments/best_auc_mac_20Mar', help='path to saved model - mac2')
parser.add_argument('--load_path_both_k', type=str, default='experiments/best_kappa_both_20Mar', help='path to saved model - od+mac1')
parser.add_argument('--load_path_both_auc', type=str, default='experiments/best_auc_both_20Mar', help='path to saved model - od+mac2')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='from pretrained weights')
parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=True, help='use tta')
parser.add_argument('--n_classes', type=int, default=5, help='number of target classes (5)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--csv_out', type=str, default='results/submission_galdran_DR_ONSITE.csv', help='path to output csv')

args = parser.parse_args()

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
    try:
        test_k, test_auc, test_acc = eval_predictions_multi(test_labels, preds_tta, probs_tta)
        print('Test Kappa: {:.4f} -- AUC: {:.4f} -- Balanced Acc: {:.4f}'.format(test_k, test_auc, test_acc))
    except:
        print('Test Kappa: {:.4f} -- AUC: {:.4f} -- Balanced Acc: {:.4f}'.format(0, 0, 0))


    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls_tta(model, test_loader):
    probs_tta = []
    prs = [0, 1]
    for p1 in prs:
        for p2 in prs:
            test_loader.dataset.transforms.transforms[1].p = p1  # pr(horizontal flip)
            test_loader.dataset.transforms.transforms[2].p = p2  # pr(vertical flip)
            # validate one epoch, note no optimizer is passed
            with torch.no_grad():
                test_preds, test_probs, test_labels = run_one_epoch_cls(test_loader, model)
                probs_tta.append(test_probs)

    probs_tta = np.mean(np.array(probs_tta), axis=0)
    preds_tta = np.argmax(probs_tta, axis=1)

    test_k, test_auc, test_acc = eval_predictions_multi(test_labels, preds_tta, probs_tta)
    print('Test Kappa: {:.4f} -- AUC: {:.4f} -- Balanced Acc: {:.4f}'.format(test_k, test_auc, test_acc))

    del model
    torch.cuda.empty_cache()
    return probs_tta, preds_tta, test_labels

def test_cls(model, test_loader):
    # validate one epoch, note no optimizer is passed
    with torch.no_grad():
        preds, probs, labels = run_one_epoch_cls(test_loader, model)
    vl_k, vl_auc, vl_acc = eval_predictions_multi(labels, preds, probs)
    print('Val. Kappa: {:.4f} -- AUC: {:.4f}'.format(vl_k, vl_auc).rstrip('0'))

    del model
    torch.cuda.empty_cache()
    return probs, preds, labels

if __name__ == '__main__':
    '''
    Inference of DR grading on fundus images, online challenge version
    Example:
    python test.py --tta True --csv_out results/submission_galdran_DR_ONSITE.csv
    '''
    data_path = 'data'
    path_subm_csv = 'results/Challenge1_upload.csv'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name

    load_path_od_k = args.load_path_od_k
    load_path_od_auc = args.load_path_od_auc

    load_path_mac_k = args.load_path_mac_k
    load_path_mac_auc = args.load_path_mac_auc

    load_path_both_k = args.load_path_both_k
    load_path_both_auc = args.load_path_both_auc

    pretrained = args.pretrained
    bs = args.batch_size
    csv_test_od = args.csv_test_od
    csv_test_mac = args.csv_test_mac
    n_classes = args.n_classes
    tta = args.tta
    csv_out = args.csv_out

    ####################################################################################################################
    # build results for od-centered with OD_k model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_od_k, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_od,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_od_k, preds_od_k, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_od_k, preds_od_k, labels = test_cls(model, test_loader)
    ####################################################################################################################
    # build results for od-centered with OD_auc model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_od_auc, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_od,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_od_auc, preds_od_auc, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_od_auc, preds_od_auc, labels = test_cls(model, test_loader)
    ####################################################################################################################
    # build results for od-centered with both_k model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_both_k, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_od,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_od_both_k, preds_od_both_k, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_od_both_k, preds_od_both_k, labels = test_cls(model, test_loader)
    ####################################################################################################################
    # build results for od-centered with both_auc model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_both_auc, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_od,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_od_both_auc, preds_od_both_auc, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_od_both_auc, preds_od_both_auc, labels = test_cls(model, test_loader)

    ####################################################################################################################
    mean_probs_od = 0.30*probs_od_k + 0.30*probs_od_auc + 0.20*probs_od_both_k + 0.20*probs_od_both_auc
    preds_od = np.argmax(mean_probs_od, axis=1)
    df_od = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_od), columns=['image_id', 'preds'])
    ####################################################################################################################

    ####################################################################################################################
    ####################################################################################################################

    # build results for macula-centered with MAC_k model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_mac_k, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_mac,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_mac_k, preds_mac_k, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_mac_k, preds_mac_k, labels = test_cls(model, test_loader)
    ####################################################################################################################
    # build results for macula-centered with MAC_auc model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_mac_auc, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_mac,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_mac_auc, preds_mac_auc, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_mac_auc, preds_mac_auc, labels = test_cls(model, test_loader)
    ####################################################################################################################
    # build results for macula-centered with both_k model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_both_k, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_mac,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_mac_both_k, preds_mac_both_k, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_mac_both_k, preds_mac_both_k, labels = test_cls(model, test_loader)
    ####################################################################################################################
    # build results for macula-centered with both_auc model
    print('* Instantiating model {}, pretrained={}'.format(model_name, pretrained))
    model, mean, std = get_arch(model_name, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_both_auc, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_mac,  batch_size=bs, mean=mean, std=std)

    if tta:
        probs_mac_both_auc, preds_mac_both_auc, labels = test_cls_tta_dihedral(model, test_loader, n=4)
    else:
        probs_mac_both_auc, preds_mac_both_auc, labels = test_cls(model, test_loader)
    ####################################################################################################################
    mean_probs_mac = 0.30 * probs_mac_k + 0.30 * probs_mac_auc + 0.20 * probs_mac_both_k + 0.20 * probs_mac_both_auc
    preds_mac = np.argmax(mean_probs_mac, axis=1)
    df_mac = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_mac), columns=['image_id', 'preds'])


    ####################################################################################################################

    df_od['image_id'] = df_od['image_id'].apply(lambda x: x.split('/')[-1][:6])
    df_mac['image_id'] = df_mac['image_id'].apply(lambda x: x.split('/')[-1][:6])
    df_all = pd.concat([df_od, df_mac], axis=0)

    # store final submission
    df_subm = pd.read_csv(path_subm_csv)

    submission = pd.merge(df_subm, df_all, on="image_id")
    submission.drop(['DR_Level'], axis=1, inplace=True)
    submission.columns = ['image_id', 'DR_Level']
    submission.to_csv(csv_out, index=False)

