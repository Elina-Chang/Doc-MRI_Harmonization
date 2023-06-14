import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from modified_01.prepare_data import *


class ImageDataset(Dataset):
    def __init__(self, transforms_=None, unaligned=True):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.GE_2dData_list, self.Philips_2dData_list = store2dData2list()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # print(index % len(self.GE_2dData_list))

        # image = self.images[index]
        # label = self.labels[index]
        #
        # image = torch.from_numpy(self.images[index]).type(torch.FloatTensor)
        # label = torch.from_numpy(self.labels[index]).type(torch.FloatTensor)
        #
        item_A = self.transform(DataPreprocessing(self.GE_2dData_list[index % len(self.GE_2dData_list)]))
        item_A = item_A.type(torch.FloatTensor)
        # print(item_A.shape)
        # plt.imshow(item_A[0],cmap="gray")
        # plt.show()

        if self.unaligned:
            item_B = self.transform(
                DataPreprocessing(self.Philips_2dData_list[random.randint(0, len(self.Philips_2dData_list) - 1)]))
        else:
            item_B = self.transform(DataPreprocessing(self.Philips_2dData_list[index % len(self.Philips_2dData_list)]))
        item_B = item_B.type(torch.FloatTensor)

        # print(item_B.shape)
        # plt.imshow(item_B[0],cmap="gray")
        # plt.show()

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.GE_2dData_list), len(self.Philips_2dData_list))
