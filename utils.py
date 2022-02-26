import torch,os
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, lr_scheduler,Adam
from dataset3d import PetDataset
import csv
from mean import get_mean_std
from opts import parse_opts

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.result_path = opt.root_path / opt.result_path
        os.makedirs(opt.result_path,exist_ok=True)

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 1 #3-》1 通道数 预训练网络通道数为3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return opt

def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)

def resume_model(resume_path, arch, model):  # 加载模型
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_train_utils(resume_path, optimizer, scheduler):  # 加载begin_epoch optimizer scheduler
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler

def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

def get_train_utils(opt, model_parameters):
    train_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    noise_path = opt.root_path/"train"/opt.noise_path
    clean_path = opt.root_path/"train"/opt.clean_path
    dataset = PetDataset(noise_path, clean_path)
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, **train_kwargs)
    train_logger = Logger(opt.result_path / 'train.log',
                          ['epoch', 'loss', 'ssim', 'psnr', 'lr'])
    train_batch_logger = Logger(
        opt.result_path / 'train_batch.log',
        ['epoch', 'batch', 'iter', 'loss', 'ssim', 'psnr', 'lr'])

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    optimizer = SGD(model_parameters,    #修改此部分的参数 载入模型之后，指定训练微调的参数
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)

    # optimizer = Adam(model_parameters,    #修改此部分的参数 载入模型之后，指定训练微调的参数
    #                 lr=opt.learning_rate,
    #                 weight_decay=opt.weight_decay)

    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    if opt.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.plateau_patience)
    else:
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             opt.multistep_milestones)
    return (train_loader, train_logger, train_batch_logger,
            optimizer, scheduler)


def get_val_utils(opt):
    val_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    noise_path = opt.root_path / "val" / opt.noise_path
    clean_path = opt.root_path / "val" / opt.clean_path
    dataset = PetDataset(noise_path, clean_path)
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, **val_kwargs)
    val_logger = Logger(opt.result_path / 'val.log',
                        ['epoch', 'loss', 'ssim','psnr'])


    return val_loader, val_logger


def get_inference_utils(opt): #返回inference的loader 返回的是推断的结果 和 实际的结果
    test_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    clean_path = opt.root_path / "test" / opt.clean_path
    infer_clean_path = opt.result /"test"/opt.noise_path
    dataset = PetDataset(infer_clean_path, clean_path)
    inference_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, **test_kwargs)
    return inference_loader


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)

