import torch
from torch import nn
import torch.backends.cudnn as cudnn
import random,time
import numpy as np

from utils import (AverageMeter, resume_model, get_train_utils, get_val_utils, get_opt, get_lr, save_checkpoint)
from model import (generate_model, load_pretrained_model, load_pretrained_model2denoise,
                   get_fine_tuning_parameters)
from metrics import calculate_ssim_psnr

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_tensor_type(torch.FloatTensor)

def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
               ):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ssimes = AverageMeter()
    psnres = AverageMeter()
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        inputs, targets = inputs.to(device).to(torch.float32), targets.to(device, non_blocking=True).to(torch.float32)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        ssim,psnr = calculate_ssim_psnr(outputs, targets) #改成ssim，psnr

        losses.update(loss.item(), inputs.size(0))
        ssimes.update(ssim, inputs.size(0))
        psnres.update(psnr, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'ssim': ssimes.val,
                'psnr': psnres.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'SSIM {ssim.val:.3f} ({ssim.avg:.3f})\t'
              'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=losses,
                                                         ssim=ssimes,
                                                         psnr = psnres))
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'ssim': ssimes.avg,
            'psnr': psnres.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/ssim', ssimes.avg, epoch)
        tb_writer.add_scalar('train/psnr', psnres.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch) # 源代码 tb_writer.add_scalar('train/lr', accuracies.avg, epoch) ???

def val_epoch(epoch,
              data_loader,
              model,
              criterion,
              device,
              logger,
              tb_writer=None,
              ):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ssimes = AverageMeter()
    psnres = AverageMeter()
    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            inputs, targets = inputs.to(device).to(torch.float32), targets.to(device, non_blocking=True).to(
                torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            ssim,psnr = calculate_ssim_psnr(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            ssimes.update(ssim, inputs.size(0))
            psnres.update(psnr, inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'SSIM {ssim.val:.3f} ({ssim.avg:.3f})\t'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'.format(epoch,
                                                                  i + 1,
                                                                  len(data_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time,
                                                                  loss=losses,
                                                                  ssim=ssimes,
                                                                  psnr=psnres))

    if logger is not None:
        logger.log({'epoch': epoch, 'loss': losses.avg, 'ssim': ssimes.avg, 'psnr': psnres.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/ssim', ssimes.avg, epoch)
        tb_writer.add_scalar('val/psnr', psnres.avg, epoch)

    return losses.avg,ssimes.avg

if __name__ == '__main__':
    opt = get_opt()

    # 固定随机种子
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    #定义和加载模型
    model = generate_model(opt)
    model.to(opt.device)
    if opt.pretrain_path:  # 存在预训练模型 载入
        if opt.model == "resUnet3D":
            model = load_pretrained_model2denoise(model, opt.pretrain_path)
        else:
            model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                          opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module)  # 需要修改参数 TODO
    else:
        parameters = model.parameters()
    print(model)
    #定义数据集和训练参数
    criterion = nn.MSELoss().to(device)  # MSE损失函数

    (train_loader, train_logger, train_batch_logger,
     optimizer, scheduler) = get_train_utils(opt, parameters)
    val_loader, val_logger = get_val_utils(opt)
    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    val_loss = None
    val_ssim = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            current_lr = get_lr(optimizer)
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, )

            if i % opt.checkpoint == 0 :
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)

        if not opt.no_val:
            prev_val_loss,ssim = val_epoch(i, val_loader, model, criterion,
                                      opt.device, val_logger, tb_writer)
            if not val_ssim or ssim>val_ssim or (ssim==val_ssim and prev_val_loss<val_loss):  #保存最好的模型
                val_ssim = ssim
                val_loss = prev_val_loss
                save_file_path = opt.result_path / 'best.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                scheduler)
                print("val_loss:",val_loss)
                print("val_ssim",val_ssim)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            scheduler.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler.step(prev_val_loss)