""""
To achieve the prediction.
"""""
import SimpleITK as sitk
import numpy as np
import glob as glob
import os
import torch

from newline_pipe_1.models import Generator


def get_data_path(root_dir=r"G:\PhD\Data_renji\GE"):
    case_path_list = glob.glob(root_dir + "\*")
    data_path_list = sorted([glob.glob(case_path + "\*Resample.nii") for case_path in case_path_list])
    data_path_list = np.squeeze(data_path_list).tolist()
    return data_path_list


def predict(pth_root_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\newline_pipe_1\output",
            store_root_path=r"G:\PhD\PycharmProjects\my_project\MRI_Harmonization\newline_pipe_1"):
    Hyperparameters = {"input_c_A": 1, "output_c_A": 1, "input_c_B": 1, "output_c_B": 1, "lr": 2e-4, "batch_size": 8,
                       "image_size": 280, "num_epochs": 200}

    # 初始化网络结构
    netG_A2B = Generator(input_nc=Hyperparameters["input_c_A"], output_nc=Hyperparameters["output_c_A"])
    netG_A2B = torch.nn.DataParallel(netG_A2B)
    # netG_B2A = Generator(input_nc=Hyperparameters["input_c_B"], output_nc=Hyperparameters["output_c_B"])
    # netG_B2A = torch.nn.DataParallel(netG_B2A)
    # 加载预训练好的模型参数
    key_word = Hyperparameters["batch_size"]
    netG_A2B.load_state_dict(
        torch.load(os.path.join(pth_root_path, f"netG_A2B_BatchSize={key_word}.pth")))
    # netG_B2A.load_state_dict(torch.load(f"output/train_3'_partial_data/log01/netG_B2A_BatchSize={key_word}.pth"))
    # 将模型放到GPU上
    if torch.cuda.is_available():
        netG_A2B.cuda()
        # netG_B2A.cuda()

    GE_modals_list = get_data_path(root_dir=r"G:\PhD\Data_renji\GE")
    for GE_modal in GE_modals_list:
        print(f"Predict on: {GE_modal}")
        print(GE_modal.split("\\")[4])
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
            output_array = output.data.cpu().numpy()

            # # 检查一下从网络出来的output
            # # 检查极值和平均值
            # print(f"output's min:{output_array.min()}"
            #       f"\toutput's max:{output_array.max()}"
            #       f"\toutput's mean:{output_array.mean()}")
            # # 检查图像
            # import matplotlib.pyplot as plt
            # plt.imshow(output_array,cmap="gray")
            # plt.show()

            output_list.append(output.data.cpu().numpy())
        output4store = np.stack(output_list, axis=0)

        # # the solution to a little bug through .CopyInformation() part.
        # from newline_pipe_1.prepare_data import crop_center3D
        # output4store = crop_center3D(output4store, 342, 342)

        store_img = sitk.GetImageFromArray(output4store)
        store_img.CopyInformation(img)
        print("store image's resolution:", store_img.GetSpacing())
        store_path = os.path.join(store_root_path, "model_prediction\predictions_on_GE_t2sag", GE_modal.split("\\")[4])
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        sitk.WriteImage(store_img, os.path.join(store_path, "t2sag_PhilipsStyle.nii"))


if __name__ == "__main__":
    predict()
    pass
