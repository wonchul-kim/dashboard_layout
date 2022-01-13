import datetime
import os
import time
import os.path as osp
import sys 

sys.path.append(osp.join(osp.dirname(__file__), 'src'))

import json
import torch
import torch.utils.data
from torch import nn
import torchvision

from coco_utils import get_coco
import pandas as pd
import presets
import utils
import argparse
from losses import auxiliar_loss, DiceLoss
from models.torchvision_models import torchvision_models
from models.unetpp import UNet, NestedUNet
import datetime
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

def get_dataset(dir_path, dataset_type, mode, transform, num_classes):
    paths = {
        "coco": (dir_path, get_coco, num_classes),
    }

    ds_path, ds_fn, num_classes = paths[dataset_type]
    ds = ds_fn(ds_path, mode=mode, transforms=transform, num_classes=num_classes)
    return ds, num_classes


def get_transform(train, base_size, crop_size):
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def evaluate(model, loss_fn, _loss_fn, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    losses = 0
    with torch.no_grad():
        for image, target, _ in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

        if _loss_fn == 'aux' or _loss_fn == 'None':
            loss = loss_fn(output, target)
        else:
            loss = loss_fn(output['out'], target)
    
            output = output['out']

            losses += loss.detach().cpu().item()
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat, losses

def train_one_epoch(model, loss_fn, _loss_fn, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target, fn in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        output = model(image)

        if _loss_fn == 'aux' or _loss_fn == 'None':
            loss = loss_fn(output, target)
        else:
            loss = loss_fn(output['out'], target)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    
    return loss.item()

def main(args, socketio=None):

    info_weights = {'best': None, 'last': None}
    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path, args.dataset_type, "train", get_transform(True, args.base_imgsz, args.crop_imgsz), args.num_classes)
    dataset_test, _ = get_dataset(args.data_path, args.dataset_type, "val", get_transform(False, args.base_imgsz, args.crop_imgsz), args.num_classes)


    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.num_workers,
        collate_fn=utils.collate_fn)

    if args.loss == 'None' or args.loss == 'aux_loss':
        loss_fn = auxiliar_loss
    elif args.loss == 'DiceLoss':
        loss_fn = DiceLoss(args.num_classes)

    if args.model_name == 'deeplabv3_resnet101':
        model = torchvision_models(args.model_name, args.pretrained, args.loss, num_classes)
        model.backbone = torch.nn.DataParallel(model.backbone, device_ids=args.device_ids)

    model.to(device)

    model = model

    if args.model_name == 'deeplabv3_resnet101':
        params_to_optimize = [
            {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        ]
    else:
        params_to_optimize = [
            {"params": [p for p in model.parameters() if p.requires_grad]},
        ]

    if args.loss == 'aux_loss':
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})


    optimizer = torch.optim.SGD(params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum,
                              nesterov=args.nesterov, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.num_epochs)) ** 0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print(">>> Loaded the model: ", args.checkpoint)

    start_time = time.time()
    min_loss = 999
    for epoch in range(args.start_epoch, args.num_epochs):
        train_one_epoch(model, loss_fn, args.loss, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        confmat, losses_val = evaluate(model, loss_fn, args.loss, data_loader_test, device=device, num_classes=num_classes)
        if socketio != None:
            socketio.emit("trainData", {'trainLoss': losses_val})

        print(confmat)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }

        if min_loss > losses_val:
            min_loss = losses_val
            utils.save_on_master(checkpoint,
                        osp.join(args.weights_path, '{}_checkpoint_best.pth'.format(args.model_name)))
            torch.save(model, os.path.join(args.weights_path, '{}_checkpoint_best.pt'.format(args.model_name)))
            info_weights['best'] = epoch
            print(">>> Saved the best model .......! ")
        utils.save_on_master(checkpoint,
                    osp.join(args.weights_path, '{}_checkpoint_last.pth'.format(args.model_name)))
        torch.save(model, os.path.join(args.weights_path, '{}_checkpoint_last.pt'.format(args.model_name)))
        info_weights['last'] = epoch

        with open(osp.join(args.weights_path, 'info.txt'), 'w') as f:
            f.write('best: {}'.format(info_weights['best']))
            f.write('\n')
            f.write('last: {}'.format(info_weights['last']))
            f.write('\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser_torchvision(add_help=True):
    
    parser = argparse.ArgumentParser(description='Segmentation', add_help=add_help)
    # parser.add_argument("--project_name", default="NONE")
    parser.add_argument('--data-path', default='/home/wonchul/HDD/datasets/projects/interojo/3rd_poc_/coco_datasets_good/react_bubble_damage_print_dust', help='dataset path')
    # parser.add_argument('--data-path', default='/home/nvadmin/wonchul/mnt/HDD/datasets/projects/interojo/3rd_poc_/coco_datasets_good/react_bubble_damage_print_dust', help='dataset path')
    parser.add_argument('--dataset-type', default='coco', help='dataset name')
    parser.add_argument('--model-name', default='deeplabv3_resnet101', help='model name')
    parser.add_argument("--pretrained", default=True)
    parser.add_argument('--loss', default='DiceLoss', help='loss type: None, BCEDiceLoss, aux_loss, DiceLoss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--device-ids', default='0,1', help='gpu device ids')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-epochs', default=131, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--base-imgsz', default=80, type=int, help='base image size')
    parser.add_argument('--crop-imgsz', default=80, type=int, help='base image size')

    parser.add_argument('--num-workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./outputs/train', help='path where to save')
    parser.add_argument('--checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--opt', default='SGD', type=str, help='optimizers')
    parser.add_argument('--nesterov', default=False, help='nesterov')

    return parser


if __name__ == "__main__":
    args = get_args_parser_torchvision().parse_args()

    args.device_ids = list(map(int, args.device_ids.split(',')))
    args.num_classes = len(list(osp.split(osp.splitext(args.data_path)[0])[-1].split('_'))) + 1

    now = datetime.datetime.now()
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            args.output_dir = os.path.join(args.output_dir, 'exp1')
            os.makedirs(args.output_dir)
        else:
            folders = os.listdir(args.output_dir)
            args.output_dir = os.path.join(args.output_dir, 'exp' + str(len(folders) + 1))
            os.makedirs(args.output_dir)

        output_path = args.output_dir
        args.date = str(datetime.datetime.now())
        utils.mkdir(osp.join(output_path, 'cfg'))
        weights_path = osp.join(output_path, 'weights')
        utils.mkdir(weights_path)
    args.wegiths_path = weights_path 

    with open(osp.join(output_path, 'cfg/config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(args)

    main(args)
