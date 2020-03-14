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
parser.add_argument('--csv_test_q', type=str, default='data/test_q.csv', help='path to test OD data csv')
parser.add_argument('--model_name_quality', type=str, default='resnet50_sws', help='selected architecture')
parser.add_argument('--model_name_artifact', type=str, default='resnet50', help='selected architecture')
parser.add_argument('--model_name_field_def', type=str, default='resnext50', help='selected architecture')
parser.add_argument('--model_name_clarity', type=str, default='resnext50', help='selected architecture')
parser.add_argument('--load_path_quality', type=str, default='experiments/best_quality_16Mar', help='path to saved model - quality')
parser.add_argument('--load_path_artifact', type=str, default='experiments/best_artifact_16Mar', help='path to saved model - artifact')
parser.add_argument('--load_path_field_def', type=str, default='experiments/best_field_def_16Mar', help='path to saved model - field def')
parser.add_argument('--load_path_clarity', type=str, default='experiments/best_clarity_16Mar', help='path to saved model - clarity')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='from pretrained weights')
parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=True, help='use tta')
parser.add_argument('--n_classes', type=int, default=5, help='number of target classes (5)')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--csv_out', type=str, default='results/submission_quality_galdran_11Mar.csv', help='path to output csv')

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
    # test_k, test_auc, test_acc = eval_predictions_multi(test_labels, preds_tta, probs_tta)
    # print('Test Kappa: {:.4f} -- AUC: {:.4f} -- Balanced Acc: {:.4f}'.format(test_k, test_auc, test_acc))

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
    model_name_quality = args.model_name_quality
    model_name_artifact = args.model_name_artifact
    model_name_field_def = args.model_name_field_def
    model_name_clarity = args.model_name_clarity
    load_path_quality = args.load_path_quality
    load_path_artifact = args.load_path_artifact
    load_path_field_def = args.load_path_field_def
    load_path_clarity = args.load_path_clarity
    pretrained = args.pretrained
    bs = args.batch_size
    csv_test_q = args.csv_test_q
    n_classes = args.n_classes
    tta = args.tta
    csv_out = args.csv_out

    ####################################################################################################################
    # build results for QUALITY model
    n_classes = 2
    print('* Instantiating QUALITY model {}, pretrained={}'.format(model_name_quality, pretrained))
    model, mean, std = get_arch(model_name_quality, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_quality, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_q,  batch_size=bs, mean=mean, std=std)
    if tta:
        probs_q, preds_q, labels = test_cls_tta_dihedral(model, test_loader, n=3)
    else:
        probs_q, preds_q, labels = test_cls(model, test_loader)
    df_quality = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_q), columns=['image_id', 'Overall quality'])
    # df_quality.to_csv('quality.csv', index=False)
    ####################################################################################################################
    # build results for ARTIFACT model
    n_classes = 6
    print('* Instantiating ARTIFACT model {}, pretrained={}'.format(model_name_artifact, pretrained))
    model, mean, std = get_arch(model_name_artifact, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_artifact, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_q, batch_size=bs, mean=mean, std=std)

    if tta:
        probs_a, preds_a, labels = test_cls_tta_dihedral(model, test_loader, n=3)
    else:
        probs_a, preds_a, labels = test_cls(model, test_loader)
    df_artifact = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_a), columns=['image_id', 'Artifact'])
    def map_label_art(label):
        if label == 0: return 0
        elif label == 1: return 1
        elif label == 2: return 4
        elif label == 3: return 6
        elif label == 4: return 8
        else: return 10
    df_artifact['Artifact'] = df_artifact['Artifact'].apply(map_label_art)
    # df_artifact.to_csv('artifact.csv', index=False)
    ####################################################################################################################
    # build results for CLARITY model
    n_classes = 5
    print('* Instantiating CLARITY model {}, pretrained={}'.format(model_name_clarity, pretrained))
    model, mean, std = get_arch(model_name_clarity, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_clarity, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_q, batch_size=bs, mean=mean, std=std)

    if tta:
        probs_c, preds_c, labels = test_cls_tta_dihedral(model, test_loader, n=3)
    else:
        probs_c, preds_c, labels = test_cls(model, test_loader)
    df_clarity = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_c), columns=['image_id', 'Clarity'])
    def map_label_clarity(lab):
        if lab == 0: return 1
        elif lab == 1: return 4
        elif lab == 2: return 6
        elif lab == 3: return 8
        else: return 10
    df_clarity['Clarity'] = df_clarity['Clarity'].apply(map_label_clarity)
    # df_clarity.to_csv('clarity.csv', index=False)
    ####################################################################################################################
    # build results for FIELD DEFINITION model
    n_classes = 5
    print('* Instantiating FIELD DEFINITION model {}, pretrained={}'.format(model_name_field_def, pretrained))
    model, mean, std = get_arch(model_name_field_def, pretrained=pretrained, n_classes=n_classes)

    model, stats = load_model(model, load_path_field_def, device='cpu')
    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    test_loader = get_test_loader(csv_path_test=csv_test_q, batch_size=bs, mean=mean, std=std)

    if tta:
        probs_f, preds_f, labels = test_cls_tta_dihedral(model, test_loader, n=3)
    else:
        probs_f, preds_f, labels = test_cls(model, test_loader)
    df_field_def = pd.DataFrame(zip(list(test_loader.dataset.im_list), preds_f), columns=['image_id', 'Field definition'])
    def map_label_fd(label):
        if label == 0: return 1
        elif label == 1: return 4
        elif label == 2: return 6
        elif label == 3: return 8
        else: return 10
    df_field_def['Field definition'] = df_field_def['Field definition'].apply(map_label_fd)
    # df_field_def.to_csv('field_def.csv', index=False)



    from functools import reduce
    submission = reduce(lambda x, y: pd.merge(x, y, on='image_id'), [df_quality, df_artifact, df_clarity, df_field_def])
    submission['image_id'] = submission['image_id'].apply(lambda x: x.split('/')[-1][:-4])
    submission = submission[['Overall quality', 'Artifact', 'Clarity', 'Field definition', 'image_id']]
    submission.to_csv(csv_out, index=False)


