import torch.utils.data
import numpy as np
import torch
import torch.utils.data
from torchvision import transforms
import random
from PIL import Image
import rasterio
import torchvision.transforms.functional as F


class FloodTrainData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, item):
        item = self.data_list[item]
        base_image = np.nan_to_num(rasterio.open(item[0]).read())
        # Normalize S1 imagery
        base_image = np.clip(base_image, -50, 1)
        base_image = (base_image + 50) / 51

        # Load mask
        mask = rasterio.open(item[1]).read()
        mask[mask == -1] = 255
        base_image, mask = base_image.copy(), mask.copy()
        base_im_0 = Image.fromarray(base_image[0])
        base_im_1 = Image.fromarray(base_image[1])
        mask = Image.fromarray(mask.squeeze())

        i, j, h, w = transforms.RandomCrop.get_params(base_im_0, (256, 256))

        base_im_0 = F.crop(base_im_0, i, j, h, w)
        base_im_1 = F.crop(base_im_1, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        if random.random() > 0.5:
            im1 = F.hflip(base_im_0)
            im2 = F.hflip(base_im_1)
            label = F.hflip(mask)
        if random.random() > 0.5:
            im1 = F.vflip(base_im_0)
            im2 = F.vflip(base_im_1)
            label = F.vflip(mask)

        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
        # im = torch.stack([transforms.ToTensor()(base_im_0).squeeze(), transforms.ToTensor()(base_im_1).squeeze()])
        im = torch.stack(
            [transforms.ToTensor()(np.array(base_im_0)).squeeze(),
             transforms.ToTensor()(np.array(base_im_1)).squeeze()])
        im = norm(im)
        label = transforms.ToTensor()(np.array(mask)).squeeze()
        return im, label, [str(item[0]), str(item[1])]

    def __len__(self):
        return len(self.data_list)


class FloodValidData(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, item):
        item = self.data_list[item]
        base_image = np.nan_to_num(rasterio.open(item[0]).read())
        # Normalize S1 imagery
        base_image = np.clip(base_image, -50, 1)
        base_image = (base_image + 50) / 51

        # Load mask
        mask = rasterio.open(item[1]).read()
        mask[mask == -1] = 255

        im, label = base_image.copy(), mask.copy()
        norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

        # convert to PIL for easier transforms
        base_im_0 = Image.fromarray(im[0])
        base_im_1 = Image.fromarray(im[1])
        label = Image.fromarray(label.squeeze())

        im = torch.stack(
            [transforms.ToTensor()(np.array(base_im_0)).squeeze(),
             transforms.ToTensor()(np.array(base_im_1)).squeeze()])
        im = norm(im)

        label = transforms.ToTensor()(np.array(label)).squeeze()

        return im, label, [str(item[0].parts[-1]), str(item[1].parts[-1])]

    def __len__(self):
        return len(self.data_list)

