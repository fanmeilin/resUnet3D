import torch
from torch import nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random,time,os
import numpy as np
import SimpleITK as sitk
from metrics import calculate_ssim_psnr
from utils import resume_model,get_opt, get_inference_utils, AverageMeter
from model import (generate_model, load_pretrained_model, load_pretrained_model2denoise,
                   get_fine_tuning_parameters)

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def save_img(save_path, data):
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, save_path)

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    data = data[np.newaxis, np.newaxis, :, :, :]
    return data

def inference_evaluate(data_loader,
              criterion,
              device,
              ):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ssimes = AverageMeter()
    psnres = AverageMeter()
    end_time = time.time()

    with torch.no_grad():
        for i, (outputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            targets = targets.to(device, non_blocking=True)
            outputs = outputs.to(device, non_blocking=True)
            loss = criterion(outputs, targets)
            ssim,psnr = calculate_ssim_psnr(outputs, targets)
            losses.update(loss.item(), targets.size(0))
            ssimes.update(ssim, targets.size(0))
            psnres.update(psnr, targets.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'SSIM {ssim.val:.3f} ({ssim.avg:.3f})\t'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'.format(i + 1,
                                                                  len(data_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time,
                                                                  loss=losses,
                                                                  ssim=ssimes,
                                                                  psnr=psnres))
    print(f'loss: {losses.avg}, ssim:{ssimes.avg} , psnr: {psnres.avg}')

if __name__ == '__main__':
    opt = get_opt()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 固定随机种子
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    model = generate_model(opt)
    model.to(device)
    if opt.pretrain_path: #存在预训练模型 载入
        if opt.model == "resUnet3D":
            model = load_pretrained_model2denoise(model, opt.pretrain_path)
        else:
            model = load_pretrained_model(model, opt.pretrain_path, opt.model,
                                      opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    if opt.pretrain_path:
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_module) #需要修改参数
    else:
        parameters = model.parameters()
    criterion = nn.MSELoss().to(device)  # MSE损失函数
    if opt.inference and opt.test_path is not None:
        with torch.no_grad():
            for item in tqdm(os.listdir(opt.test_path)):
                item_path = os.path.join(opt.test_path, item)
                data = read_img(item_path)
                data = np.array(data, 'float32')
                data -= data.mean()  # 标准化
                data /= data.std()
                data = torch.FloatTensor(data).to(device)
                output = model(data)
                result_test = opt.result_path /"test"/opt.noise_path
                os.makedirs(result_test, exist_ok=True)
                save_img(os.path.join(result_test, item), output.cpu().detach().numpy()) #保存数据


        inference_loader = get_inference_utils(opt)
        inference_evaluate(inference_loader,criterion,device)