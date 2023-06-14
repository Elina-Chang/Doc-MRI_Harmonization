""""
To achieve the prediction.
"""""
import SimpleITK as sitk
import numpy as np
import glob as glob
import os
import torch

from vanilla_cycleGAN.models import Generator
from Tools.normalization1.generate_clip_value import generate_clip_value


def get_data_path(root_dir=r"G:\PhD\Data_renji\Data_3T_Norm1\GE_3T_Norm1"):
    case_path_list = glob.glob(root_dir + "\*")
    data_path_list = sorted([glob.glob(case_path + "\*t2sag.nii") for case_path in case_path_list])
    data_path_list = np.squeeze(data_path_list).tolist()
    return data_path_list[1:2]


def normalization_after_clip(data3D):
    clip_value = generate_clip_value(data3D)
    data3D[data3D < clip_value["bottom"]] = clip_value["bottom"]
    data3D[data3D > clip_value["top"]] = clip_value["top"]
    data3D = (data3D - data3D.min()) / (data3D.max() - data3D.min())
    return clip_value, data3D


def predict(pth_root_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\vanilla_cycleGAN"
                          r"\output\Norm1_41GE_and_75Philips\checkpoints\100"):
    Hyperparameters = {"input_c_A": 1, "output_c_A": 1, "input_c_B": 1, "output_c_B": 1, "batch_size": 1,
                       "num_epochs": 100}

    # 初始化网络结构
    netG_A2B = Generator(input_nc=Hyperparameters["input_c_A"], output_nc=Hyperparameters["output_c_A"])
    netG_A2B = torch.nn.DataParallel(netG_A2B)
    # netG_B2A = Generator(input_nc=Hyperparameters["input_c_B"], output_nc=Hyperparameters["output_c_B"])
    # netG_B2A = torch.nn.DataParallel(netG_B2A)
    # 加载预训练好的模型参数
    batch_size = Hyperparameters["batch_size"]
    epoch = Hyperparameters["num_epochs"]
    netG_A2B.load_state_dict(
        torch.load(os.path.join(pth_root_path, f"netG_A2B_BatchSize={batch_size}_epoch={epoch}.pth")))
    # netG_B2A.load_state_dict(torch.load(f"output/train_3'_partial_data/log01/netG_B2A_BatchSize={key_word}.pth"))
    # 将模型放到GPU上
    if torch.cuda.is_available():
        netG_A2B.cuda()
        # netG_B2A.cuda()

    GE_modals_list = get_data_path(root_dir=r"G:\PhD\Data_renji\Data_3T_Norm1\GE_3T_Norm1")
    for GE_modal in GE_modals_list:
        print(f"Predict on: {GE_modal}")
        print(GE_modal.split("\\")[5])
        img = sitk.ReadImage(GE_modal)
        # print(img.GetSpacing())
        data = sitk.GetArrayFromImage(img)
        data = (data - data.min()) / (data.max() - data.min())  # Rescale to [0,1]
        print(data.min(), data.max())
        # import matplotlib.pyplot as plt
        # plt.imshow(data[0], cmap="gray")
        # plt.show()

        output_list = []
        for slice_index, slice in enumerate(data):
            # 检查一下slice_index
            # print(slice_index)
            # Put into the model
            # convert to tensor and unsqueeze to [B,C,H,W]（这里为[1,1,H,W]）
            slice = torch.unsqueeze(torch.tensor(slice, device="cuda", requires_grad=False, dtype=torch.float), dim=0)
            slice = torch.unsqueeze(slice, dim=0)
            output = netG_A2B(slice)
            # squeeze掉多余的维度，重新变回[H,W]
            output = torch.squeeze(output)
            output_array = output.data.cpu().numpy()  # [1:-1, 1:-1]

            # # 检查一下从网络出来的output
            # # 检查极值和平均值
            # print(f"output's min:{output_array.min()}"
            #       f"\toutput's max:{output_array.max()}"
            #       f"\toutput's mean:{output_array.mean()}")
            # # 检查图像
            # import matplotlib.pyplot as plt
            # plt.imshow(output_array,cmap="gray")
            # plt.show()

            output_list.append(output_array)
        output4store = np.stack(output_list, axis=0)
        clip_value, output4store = normalization_after_clip(output4store)  # normalization1
        print(clip_value)

        store_img = sitk.GetImageFromArray(output4store)
        store_img.CopyInformation(img)
        print("store image's resolution:", store_img.GetSpacing())
        store_path = os.path.join(r"G:\PhD\StyleTransferPredictions\vanilla_cycleGAN\predictions",
                                  GE_modal.split("\\")[5])
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        sitk.WriteImage(store_img, os.path.join(store_path, "t2sag_PhilipsStyle.nii.gz"))


if __name__ == "__main__":
    predict()
