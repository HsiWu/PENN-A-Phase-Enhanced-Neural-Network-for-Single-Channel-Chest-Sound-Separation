import torch
import os
import random
from pathlib import Path


class My_Data(torch.utils.data.Dataset):
    def __init__(self, root_dir, heart_dir, lung_dir, complex_mask, steps):
        super(My_Data).__init__()
        self.root_dir = root_dir
        self.heart_dir = heart_dir
        self.lung_dir = lung_dir
        self.complex_mask = complex_mask
        self.steps = steps
        self.heart_path = os.path.join(self.root_dir, self.heart_dir)
        self.lung_path = os.path.join(self.root_dir, self.lung_dir)
        self.heart_document_path = [f.name for f in Path(self.heart_path).iterdir() if f.is_file()]
        self.lung_document_path = [f.name for f in Path(self.lung_path).iterdir() if f.is_file()]
        heart_length = len(self.heart_document_path)
        lung_length = len(self.lung_document_path)

        self.index_heart = [random.randint(0, heart_length - 1) for _ in range(steps)]
        self.index_lung = [random.randint(0, lung_length - 1) for _ in range(steps)]

    def __getitem__(self, index):
        heart_itme_path = os.path.join(self.heart_path, self.heart_document_path[self.index_heart[index]])
        lung_item_path = os.path.join(self.lung_path, self.lung_document_path[self.index_lung[index]])
        heart_item = torch.load(heart_itme_path)[:51, :]
        lung_item = torch.load(lung_item_path)[:51, :]
        mix_item = heart_item + lung_item
        heart_real = torch.real(heart_item)
        heart_imag = torch.imag(heart_item)
        lung_real = torch.real(lung_item)
        lung_imag = torch.imag(lung_item)
        mix_real = torch.real(mix_item)
        mix_imag = torch.imag(mix_item)
        if self.complex_mask:
            K = 10
            C = 0.1

            a = mix_real * heart_real
            b = mix_imag * heart_imag
            c = mix_real * heart_imag
            d = mix_imag * heart_real
            bottom = mix_real ** 2 + mix_imag ** 2
            mask_real = (a + b) / (bottom + 1e-6)
            mask_imag = (c - d) / (bottom + 1e-6)
            exp_real_heart = torch.exp(-C * mask_real)
            exp_imag_heart = torch.exp(-C * mask_imag)

            a = mix_real * lung_real
            b = mix_imag * lung_imag
            c = mix_real * lung_imag
            d = mix_imag * lung_real
            bottom = mix_real ** 2 + mix_imag ** 2
            mask_real = (a + b) / (bottom + 1e-6)
            mask_imag = (c - d) / (bottom + 1e-6)
            exp_real_lung = torch.exp(-C * mask_real)
            exp_imag_lung = torch.exp(-C * mask_imag)

            return heart_real, heart_imag, lung_real, lung_imag, \
                   K * (1 - exp_real_heart) / (1 + exp_real_heart), K * (1 - exp_imag_heart) / (1 + exp_imag_heart), \
                   K * (1 - exp_real_lung) / (1 + exp_real_lung), K * (1 - exp_imag_lung) / (1 + exp_imag_lung), \
                   mix_real, mix_imag

        else:
            heart_abs = torch.abs(heart_item)
            SNR_heart = (heart_abs / (torch.abs(lung_item) + 1e-6)) ** 2
            lung_abs = torch.abs(lung_item)
            SNR_lung = (lung_abs / (torch.abs(heart_item) + 1e-6)) ** 2
            return heart_real, heart_imag, lung_real, lung_imag, \
                   torch.sqrt(SNR_heart / (SNR_heart + 1)), torch.sqrt(SNR_lung / (SNR_lung + 1)), \
                   mix_real, mix_imag

    def __len__(self):
        return self.steps
