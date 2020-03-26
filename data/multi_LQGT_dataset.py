import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import pdb


class MULTI_LQGTDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(MULTI_LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ_x8, self.paths_LQ_x12, self.paths_LQ_x16, self.paths_LQ_x24, self.paths_GT = None, None, None, None, None
        self.sizes_LQ_x8, self.sizes_LQ_x12, self.sizes_LQ_x16, self.sizes_LQ_x24, self.sizes_GT = None, None, None, None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb
        #pdb.set_trace()
        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])

        self.paths_LQ_x8, self.paths_LQ_x12, self.paths_LQ_x16, self.paths_LQ_x24, \
        self.sizes_LQ_x8, self.sizes_LQ_x12, self.sizes_LQ_x16, self.sizes_LQ_x24 = \
            util.get_multi_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if (self.paths_LQ_x8 and self.paths_LQ_x12 and self.paths_LQ_x16 and self.paths_LQ_x24 and self.paths_GT):
            assert len(self.paths_LQ_x8)  ==  len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ_x8), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = util.modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LQ image
        if self.paths_LQ_x8 and self.paths_LQ_x12 and self.paths_LQ_x24:
            LQ_path_x8 = self.paths_LQ_x8[index]
            LQ_path_x12 = self.paths_LQ_x12[index]
            LQ_path_x16 = self.paths_LQ_x16[index]
            LQ_path_x24 = self.paths_LQ_x24[index]
            resolution_x8 = [int(s) for s in self.sizes_LQ_x8[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            resolution_x12 = [int(s) for s in self.sizes_LQ_x12[index].split('_')
                             ] if self.data_type == 'lmdb' else None
            resolution_x16 = [int(s) for s in self.sizes_LQ_x16[index].split('_')
                              ] if self.data_type == 'lmdb' else None
            resolution_x24 = [int(s) for s in self.sizes_LQ_x24[index].split('_')
                             ] if self.data_type == 'lmdb' else None

            img_LQ_x8 = util.read_img(self.LQ_env, LQ_path_x8, resolution_x8)
            img_LQ_x12 = util.read_img(self.LQ_env, LQ_path_x12, resolution_x12)
            img_LQ_x16 = util.read_img(self.LQ_env, LQ_path_x16, resolution_x16)
            img_LQ_x24 = util.read_img(self.LQ_env, LQ_path_x24, resolution_x24)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)



            # randomly crop
            for scale,img_LQ in zip([8,12,16,24],[img_LQ_x8,img_LQ_x12,img_LQ_x16,img_LQ_x24]):
                H, W, C = img_LQ.shape
                LQ_size = GT_size // scale
                rnd_h = random.randint(0, max(0, H - LQ_size))
                rnd_w = random.randint(0, max(0, W - LQ_size))
                img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
                img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ_x8, img_LQ_x12, img_LQ_x16, img_LQ_x24, img_GT = util.augment([img_LQ_x8, img_LQ_x12, img_LQ_x16, img_LQ_x24, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        if self.opt['color']:  # change color space if necessary
            img_LQ_x8 = util.channel_convert(C, self.opt['color'],
                                          [img_LQ_x8])[0]  # TODO during val no definition
            img_LQ_x12 = util.channel_convert(C, self.opt['color'],
                                             [img_LQ_x12])[0]  # TODO during val no definition
            img_LQ_x16 = util.channel_convert(C, self.opt['color'],
                                              [img_LQ_x16])[0]  # TODO during val no definition
            img_LQ_x24 = util.channel_convert(C, self.opt['color'],
                                             [img_LQ_x24])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ_x8 = img_LQ_x8[:, :, [2, 1, 0]]
            img_LQ_x12 = img_LQ_x12[:, :, [2, 1, 0]]
            img_LQ_x16 = img_LQ_x16[:, :, [2, 1, 0]]
            img_LQ_x24 = img_LQ_x24[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ_x8 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_x8, (2, 0, 1)))).float()
        img_LQ_x12 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_x12, (2, 0, 1)))).float()
        img_LQ_x16 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_x16, (2, 0, 1)))).float()
        img_LQ_x24 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ_x24, (2, 0, 1)))).float()

        #if LQ_path_x8 or LQ_path_x12 or LQ_path_x24 is None:
            #LQ_path = GT_path
        return {'LQ_x8': img_LQ_x8, 'LQ_x12': img_LQ_x12, 'LQ_x16': img_LQ_x16, 'LQ_x24': img_LQ_x24,'GT': img_GT,
                'LQ_path_x8': LQ_path_x8,'LQ_path_x12': LQ_path_x12, 'LQ_path_x24': LQ_path_x24,'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
