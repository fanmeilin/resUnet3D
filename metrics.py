import os
import numpy
import numpy as np
import math
import torch
import SimpleITK as sitk
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt


# resnet中的acc计算
def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def calculate_ssim_psnr(outputs, targets, max_i=1.0):
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    sum_ssim = 0
    sum_psnr = 0

    with torch.no_grad():
        batch_size = targets.size(0)
        for i, out in enumerate(outputs):
            # 二维切片
            temp_ssim = 0
            temp_psnr = 0
            temp_target = targets[i]  # 当前batch中的某一个target
            temp_target = temp_target.cpu().numpy()
            out = out.cpu().numpy()
            for j in range(temp_target.shape[0]):
                temp_ssim += compare_ssim(temp_target[j, :, :], out[j, :, :], data_range=max_i)
                temp_psnr += compare_psnr(temp_target[j, :, :], out[j, :, :], data_range=max_i)

            sum_ssim += temp_ssim / temp_target.shape[0]
            sum_psnr += temp_psnr / temp_target.shape[0]  # 所有切片求平均值

    return sum_ssim / batch_size, sum_psnr / batch_size

def read_img(path, MAX_I):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    # print(data.shape)
    # print("data", data)

    data -= data.mean()  # 标准化
    data /= data.std()

    # # 归一化
    # data = (data - data.min()) / (data.max() - data.min())
    # print("data", data)

    # 根据MAX_I决定是否恢复到0-255
    data *= MAX_I
    # print("data", data)

    # return np.float64(data)
    return data


def psnr(img1, img2, MAX_I=255):
    """
    自己实现的psnr, 经验证, 与skimage.metrics中的psnr效果相同
    :param img1:
    :param img2:
    :param MAX_I:
    :return:
    """
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = numpy.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = MAX_I
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# def ssim(img1, img2, MAX_I=255):
#     """
#     pytorch实现的ssim, 只能用于2D
#     :param img1:
#     :param img2:
#     :param MAX_I:
#     :return:
#     """
#     # img1 = torch.from_numpy(np.rollaxis(img1, 2)).float().unsqueeze(0) / MAX_I
#     # img2 = torch.from_numpy(np.rollaxis(img2, 2)).float().unsqueeze(0) / MAX_I
#     img1 = Variable(torch.from_numpy(img1), requires_grad=False)
#     img2 = Variable(torch.from_numpy(img2), requires_grad=False)
#     if torch.cuda.is_available():
#         img1 = img1.cuda()
#         img2 = img2.cuda()
#     ssim_value = pytorch_ssim.ssim(img1, img2).item()
#     return ssim_value


def mse(img1, img2):
    # img1 = np.float64(img1)
    # img2 = np.float64(img2)
    # mse = numpy.mean((img1 - img2) ** 2)
    # return mse
    return np.mean((img1 - img2) ** 2, dtype=np.float64)


if __name__ == "__main__":
    FD_path = r"data3d/FDG/FD"
    LD_pos_path = r"data3d/FDG/LD_pos"
    LD_pos_REDNet10_path = r"data3d/FDG/LD_pos_REDNet10"
    LD_pos_REDNet20_path = r"data3d/FDG/LD_pos_REDNet20"
    LD_pos_REDNet30_path = r"data3d/FDG/LD_pos_REDNet30"
    LD_fly_path = r"data3d/FDG/LD_fly"
    LD_fly_REDNet20_path = r"data3d/FDG/LD_fly_REDNet20"

    sum_psnr = [0, 0, 0, 0, 0, 0, 0]
    sum_ssim = [0, 0, 0, 0, 0, 0, 0]
    sum_mse = [0, 0, 0, 0, 0, 0, 0]

    list_psnr = [[], [], [], [], [], [], []]
    list_ssim = [[], [], [], [], [], [], []]
    list_mse = [[], [], [], [], [], [], []]

    max_i = 255  # 根据像素值范围进行设置, 归一化计算则设置为1.0, 恢复为0-255灰度值则设置为255.0

    for item in os.listdir(FD_path):
        FD = read_img(os.path.join(FD_path, item), max_i)
        LD_pos = read_img(os.path.join(LD_pos_path, item), max_i)
        LD_pos_REDNet10 = read_img(os.path.join(LD_pos_REDNet10_path, item), max_i)
        LD_pos_REDNet20 = read_img(os.path.join(LD_pos_REDNet20_path, item), max_i)
        LD_pos_REDNet30 = read_img(os.path.join(LD_pos_REDNet30_path, item), max_i)
        LD_fly = read_img(os.path.join(LD_fly_path, item), max_i)
        LD_fly_REDNet20 = read_img(os.path.join(LD_fly_REDNet20_path, item), max_i)

        compute_list = [FD, LD_pos, LD_pos_REDNet10, LD_pos_REDNet20, LD_pos_REDNet30, LD_fly, LD_fly_REDNet20]
        for i, l in enumerate(compute_list):
            # # 三维直接计算
            # # sum_psnr[i] += compare_psnr(FD, l, data_range=max_i)
            # sum_psnr[i] += psnr(FD, l, MAX_I=max_i)
            # sum_ssim[i] += compare_ssim(FD, l, data_range=max_i)
            #
            # sum_mse[i] += mse(FD, l)

            # 二维切片
            temp_psnr = 0
            temp_ssim = 0
            temp_mse = 0
            for j in range(FD.shape[0]):
                # temp_psnr += compare_psnr(FD[j, :, :], l[j, :, :], data_range=max_i)
                temp_psnr += psnr(FD[j, :, :], l[j, :, :], MAX_I=max_i)
                temp_ssim += compare_ssim(FD[j, :, :], l[j, :, :], data_range=max_i)

                # s1 = FD[j, :, :]
                # s2 = l[j, :, :]
                # temp_ssim += ssim(s1[np.newaxis, np.newaxis, :], s2[np.newaxis, np.newaxis, :], MAX_I=max_i)

                temp_mse += mse(FD[j, :, :], l[j, :, :])

            sum_psnr[i] += temp_psnr / FD.shape[0]  # 所有切片求平均值
            sum_ssim[i] += temp_ssim / FD.shape[0]
            sum_mse[i] += temp_mse / FD.shape[0]

            list_psnr[i].append(temp_psnr / FD.shape[0])
            list_ssim[i].append(temp_ssim / FD.shape[0])
            list_mse[i].append(temp_mse / FD.shape[0])

    count = len(os.listdir(FD_path))
    print("PSNR values: ", np.array(sum_psnr) / count)
    print("SSIM values: ", np.array(sum_ssim) / count)
    print("MSE values: ", np.array(sum_mse) / count)

    # 可视化计算结果
    x = list(range(1, count + 1))
    color_list = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']
    label_list = ['FF', 'Fpos', 'FposR10', 'FposR20', 'FposR30', 'Ffly', 'Ffly20']

    fig1, ax1 = plt.subplots(1, 1)
    for i, item in enumerate(list_psnr):
        ax1.plot(x, item, color=color_list[i], label=label_list[i])
    ax1.set_title('PSNR Values')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Value')
    ax1.set_yticks(np.arange(0, 101, 10))
    ax1.legend()
    plt.savefig("PSNR.png")

    fig2, ax2 = plt.subplots(1, 1)
    for i, item in enumerate(list_ssim):
        ax2.plot(x, item, color=color_list[i], label=label_list[i])
    ax2.set_title('SSIM Values')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Value')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.legend()
    plt.savefig("SSIM.png")

    fig3, ax3 = plt.subplots(1, 1)
    for i, item in enumerate(list_mse):
        ax3.plot(x, item, color=color_list[i], label=label_list[i])
    ax3.set_title('MSE Values')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Value')
    # ax3.set_yticks(np.arange(0, 13000, 500))  # 255
    ax3.set_yticks(np.arange(0, 0.1, 0.002))
    ax3.legend()
    plt.savefig("MSE.png")

    # 单独查看LPOSR20和LPOSR30的PSNR
    fig4, ax4 = plt.subplots(1, 1)
    ax4.plot(x, list_psnr[3], color=color_list[3], label=label_list[3])
    ax4.plot(x, list_psnr[4], color=color_list[4], label=label_list[4])
    ax4.set_title('PNSR Values(POS R20 vs R30)')
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Value')
    ax4.set_yticks(np.arange(0, 101, 10))
    ax4.legend()
    plt.savefig("PNSR_POS_R20_R30.png")

    # 单独查看LPOSR20和LPOSR30的MSE
    fig5, ax5 = plt.subplots(1, 1)
    ax5.plot(x, list_mse[3], color=color_list[3], label=label_list[3])
    ax5.plot(x, list_mse[4], color=color_list[4], label=label_list[4])
    ax5.set_title('MSE Values(POS R20 vs R30)')
    ax5.set_xlabel('Sample')
    ax5.set_ylabel('Value')
    # ax5.set_yticks(np.arange(0, 500, 50))  # 255
    ax5.set_yticks(np.arange(0, 0.1, 0.002))
    ax5.legend()
    plt.savefig("MSE_POS_R20_R30.png")

    # 三维示例
    # FD = read_img("data3d/FDG/test/FD/I26991.nii")
    # # print(np.min(original))
    # # print(np.max(original))
    # LD_pos = read_img("data3d/FDG/test/LD_pos/I26991.nii")
    # LD_pos_10_denoised = read_img("results/test/REDNet10/LD_pos/I26991.nii")
    # LD_pos_20_denoised = read_img("results/test/REDNet20/LD_pos/I26991.nii")
    # LD_pos_30_denoised = read_img("results/test/REDNet30/LD_pos/I26991.nii")
    # LD_fly = read_img("data3d/FDG/test/LD_fly/I26991.nii")
    # LD_fly_20_denoised = read_img("results/test/REDNet20/LD_fly/I26991.nii")
    #
    # # PSNR
    # # psnrValue1 = psnr(original, contrast1)
    # # psnrValue2 = psnr(original, contrast2)
    # psnrValue0 = compare_psnr(FD, FD, data_range=255)
    # psnrValue1 = compare_psnr(FD, LD_pos, data_range=255)
    # psnrValue2 = compare_psnr(FD, LD_pos_10_denoised, data_range=255)
    # psnrValue3 = compare_psnr(FD, LD_pos_20_denoised, data_range=255)
    # psnrValue4 = compare_psnr(FD, LD_pos_30_denoised, data_range=255)
    # psnrValue5 = compare_psnr(FD, LD_fly, data_range=255)
    # psnrValue6 = compare_psnr(FD, LD_fly_20_denoised, data_range=255)
    #
    # print("psnrValue0: " + str(psnrValue0))
    # print("psnrValue1: " + str(psnrValue1))
    # print("psnrValue2: " + str(psnrValue2))
    # print("psnrValue3: " + str(psnrValue3))
    # print("psnrValue4: " + str(psnrValue4))
    # print("psnrValue5: " + str(psnrValue5))
    # print("psnrValue6: " + str(psnrValue6))
    #
    # # SSIM
    # # ssimValue = ssim(original, contrast1)
    # ssimValue0 = compare_ssim(FD, FD, data_range=255)
    # ssimValue1 = compare_ssim(FD, LD_pos, data_range=255)
    # ssimValue2 = compare_ssim(FD, LD_pos_10_denoised, data_range=255)
    # ssimValue3 = compare_ssim(FD, LD_pos_20_denoised, data_range=255)
    # ssimValue4 = compare_ssim(FD, LD_pos_30_denoised, data_range=255)
    # ssimValue5 = compare_ssim(FD, LD_fly, data_range=255)
    # ssimValue6 = compare_ssim(FD, LD_fly_20_denoised, data_range=255)
    # print("ssimValue0: " + str(ssimValue0))
    # print("ssimValue1: " + str(ssimValue1))
    # print("ssimValue2: " + str(ssimValue2))
    # print("ssimValue3: " + str(ssimValue3))
    # print("ssimValue4: " + str(ssimValue4))
    # print("ssimValue5: " + str(ssimValue5))
    # print("ssimValue6: " + str(ssimValue6))
    #
    # # MSE
    # mseValue0 = mse(FD, FD)
    # mseValue1 = mse(FD, LD_pos)
    # mseValue2 = mse(FD, LD_pos_10_denoised)
    # mseValue3 = mse(FD, LD_pos_20_denoised)
    # mseValue4 = mse(FD, LD_pos_30_denoised)
    # mseValue5 = mse(FD, LD_fly)
    # mseValue6 = mse(FD, LD_fly_20_denoised)
    # print("mseValue0: " + str(mseValue0))
    # print("mseValue1: " + str(mseValue1))
    # print("mseValue2: " + str(mseValue2))
    # print("mseValue3: " + str(mseValue3))
    # print("mseValue4: " + str(mseValue4))
    # print("mseValue5: " + str(mseValue5))
    # print("mseValue6: " + str(mseValue6))

    # 二维示例
    # original = cv2.imread(r"C:\Users\Shane\Desktop\8.jpg")[:560, :700]
    # print(original.shape)
    # contrast1 = cv2.imread(r"C:\Users\Shane\Desktop\9.jpg")
    # print(contrast1.shape)
    # contrast2 = cv2.imread(r"C:\Users\Shane\Desktop\8_1.jpg")[:560, :700]

    # psnrValue6 = compare_psnr(original, contrast1, data_range=255)
    # psnrValue7 = compare_psnr(original, contrast2, data_range=255)
    # print("psnrValue6: " + str(psnrValue6))
    # print("psnrValue7: " + str(psnrValue7))

    # ssimValue6 = compare_ssim(original, contrast1, multichannel=True)
    # ssimValue7 = compare_ssim(original, contrast2, multichannel=True)
    # print("ssimValue6: " + str(ssimValue6))
    # print("ssimValue7: " + str(ssimValue7))

    # mseValue6 = mse(original, contrast1)
    # mseValue7 = mse(original, contrast2)
    # print("mseValue6: " + str(mseValue6))
    # print("mseValue7: " + str(mseValue7))
