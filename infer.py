import numpy as np
import torch
import torchvision.transforms as T

from tqdm import tqdm
from pedestrian_metrics import get_pedestrian_metrics
from utils import set_seed
from pedes import PedesAttr
from torch.utils.data import DataLoader
from models import resnet50, fusion, classifier
from loss import BCELoss
from collections import OrderedDict

def logits4pred(logits_list):
    logits = logits_list[0]
    probs = logits.sigmoid()

    return probs, logits

def get_transform():
    height = 256
    width = 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

def valid_infer(model, valid_loader, criterion):
    model.eval()

    preds_probs = []
    preds_logits = []
    gt_list = []
    imgname_list = []
    loss_mtr_list = []

    embeddings = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits, feat = model(imgs, gt_label)

            mean = valid_logits[0].mean(dim=0)
            var = valid_logits[0].var(dim=0)

            valid_logits = [valid_logits[0] - mean / torch.sqrt((var + 1e-6))]

            embeddings.append(feat.detach().cpu().numpy())

            valid_probs, valid_logits = logits4pred(valid_logits)

            preds_probs.append(valid_probs.cpu().numpy())
            preds_logits.append(valid_logits.cpu().numpy())

            torch.cuda.synchronize()

            imgname_list.append(imgname)

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    preds_logits = np.concatenate(preds_logits, axis=0)


    return gt_label, preds_probs, imgname_list, preds_logits, loss_mtr_list

def main():

    set_seed(605)
    #specify dataset root
    imagepath = r'C:\Users\leehy\Downloads\Rethinking_of_PAR-master (1)\Rethinking_of_PAR-master\data\RAP\RAP_dataset\RAP_dataset/'
    train_tsfm, valid_tsfm = get_transform()

    train_set = PedesAttr(split='trainval', imagepath=imagepath, transform=train_tsfm,
                          target_transform=[])
    valid_set = PedesAttr(split='test', imagepath=imagepath, transform=valid_tsfm,
                          target_transform=[])

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    labels = train_set.label
    label_ratio = labels.mean(0)
    backbone = resnet50()
    c_output = 2048
    backbone = fusion(backbone)

    model = classifier(backbone, train_set.attr_num, c_output)
    model = model.cuda()

    state_dict_parallel = torch.load('RAP_sgd.pth', map_location=lambda storage, loc: storage)
    state_dict = OrderedDict()

    for k, v in state_dict_parallel.items():
        name = k.replace("module.", "")
        state_dict[name] = v

    model.load_state_dict(state_dict, strict=False)

    criterion = BCELoss(sample_weight=label_ratio, size_sum=True)
    criterion = criterion.cuda()

    valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_infer(
        model=model,
        valid_loader=valid_loader,
        criterion=criterion,
    )
    valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None)

    print(f'Evaluation on test set\n',
          'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
              valid_result.ma, np.mean(valid_result.label_f1),
              np.mean(valid_result.label_pos_recall),
              np.mean(valid_result.label_neg_recall)),
          'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
              valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
              valid_result.instance_f1))

if __name__ == '__main__':
    main()
