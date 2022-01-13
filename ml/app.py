from flask import Flask, request, jsonify, session
from flask_socketio import SocketIO, send, emit

from threading import Thread, Lock, Event 
from gevent.pywsgi import WSGIServer
from geventwebsocket.handler import WebSocketHandler

from train import *
import torch
import datetime 
import argparse
import os
import os.path as osp
import utils
import json

async_mode = None
async_mode = "threading"
# async_mode = "eventlet"
# async_mode = "gevent"

app = Flask(__name__)
app.debug = True 
app.host = 'localhost'
app.port = 5000
socketio = SocketIO(app, async_mode=async_mode, cors_allowed_origins="*")
thread_lock = Lock()
thread_stop_event = Event()


# To get train parameters **************************************************
@app.route("/mlworkflow/train-parameters", methods=["GET", "POST"])
def add_train_params():
    params = request.json

    global args
    args = save_parameters(params)

    return jsonify(params)

def save_parameters(params):
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

    args.project_name = params['project_name']
    # args.model_name = params['model_name']
    args.base_imsgz = int(params['image_size'])
    args.batch_size = int(params['batch_size'])
    args.num_epochs = int(params['number_of_epochs'])
    args.lr = float(params['learning_rate'])


    with open(osp.join(output_path, 'cfg/config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # print("*************************** ARGUMENTS ***************************")
    # print(args)
    # print("-----------------------------------------------------------------")

    return args

def background_thread_():
    """Example of how to send server generated events to clients."""
    count = 0
    args.batch_size *= 4
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
    

    try:
        epoch = 0
        losses_val = 0
        train_loss = 0
        epoch = args.start_epoch
        while not thread_stop_event.isSet() and epoch < args.num_epochs:
            train_loss = train_one_epoch(model, loss_fn, args.loss, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
            confmat, losses_val = evaluate(model, loss_fn, args.loss, data_loader_test, device=device, num_classes=num_classes)

            # print(confmat)

            # checkpoint = {
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            #     'lr_scheduler': lr_scheduler.state_dict(),
            #     'epoch': epoch,
            #     'args': args
            # }

            # if min_loss > losses_val:
            #     min_loss = losses_val
            #     utils.save_on_master(checkpoint,
            #                 osp.join(args.weights_path, '{}_checkpoint_best.pth'.format(args.model_name)))
            #     torch.save(model, os.path.join(args.weights_path, '{}_checkpoint_best.pt'.format(args.model_name)))
            #     info_weights['best'] = epoch
            #     print(">>> Saved the best model .......! ")
            # utils.save_on_master(checkpoint,
            #             osp.join(args.weights_path, '{}_checkpoint_last.pth'.format(args.model_name)))
            # torch.save(model, os.path.join(args.weights_path, '{}_checkpoint_last.pt'.format(args.model_name)))
            # info_weights['last'] = epoch

            # with open(osp.join(args.weights_path, 'info.txt'), 'w') as f:
            #     f.write('best: {}'.format(info_weights['best']))
            #     f.write('\n')
            #     f.write('last: {}'.format(info_weights['last']))
            #     f.write('\n')

            # main(args, socketio)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {}: {}  &  {} ".format(epoch, losses_val, train_loss))
            socketio.emit("trainData", {"epoch": epoch, "valLoss": losses_val, "trainLoss": train_loss})
            epoch += 1
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print("Keyboard  Interrupt")

def background_thread():
    try:
        epoch = 0
        losses_val = 0
        train_loss = 0
        while not thread_stop_event.isSet():
            socketio.emit("trainData", {"epoch": epoch, "valLoss": losses_val, "trainLoss": train_loss})
            print(train_loss)
            print(">> {}: {}  &  {} ".format(epoch, losses_val, train_loss))
            socketio.sleep(1)
            epoch += 1
            losses_val += 0.1
            train_loss += 0.2
    except KeyboardInterrupt:
        print("Keyboard  Interrupt")

# @socketio.on("connect")
@socketio.event
def connect():
    emit('responseConnection', {'connection': 'Connected'})
    print("*********************************************************************")
    print("                        C O N N E T E D ! ! !                        ")
    print("*********************************************************************")

@socketio.on('Start')
def handle_message(msg):
    print('[From dashboard] ', msg['status'])
    emit('connect', {'connection': "Connected"})
    thread_stop_event.clear()
    # with thread_lock:
    socketio.start_background_task(background_thread_)

@socketio.on('Stop')
def handle_message(msg):
    print('[From dashboard] ', msg['status'])
    emit('disconnect', {'connection': "NOT Connected"})
    thread_stop_event.set()

if __name__ == '__main__':
    socketio.run(app)

    # http_server = WSGIServer(('',5000), app, handler_class=WebSocketHandler)
    # http_server.serve_forever()