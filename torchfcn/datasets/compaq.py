import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data

class COMPAQSkinSeg(data.Dataset):
	class_names = np.array(['non-skin', 'skin'])
	mean_bgr = np.array([127, 127, 127]) # Not computed yet

	def __init__(self, root, split = 'train', trainsform=False):
		self.root = root
		self.split = split
		self._trainform = trainsform
		
		dataset_dir = osp.join(self.root, 'CompaqSkinDataset')
		self.files = collections.defaultdict(list)
		for split in ['train', 'val']:
			imgsets_file = osp.join(dataset_dir, '%s.txt'%split)
			for did in open(imgsets_file):
				did = did.strip()
				img_file, lbl_file = did
				img_file = osp.join(dataset_dir, 'skin-images/%s'%img_file)
				lbl_file = osp.join(dataset_dir, 'masks/%s'%lbl_file)

				self.files[split].append({
					'img': img_file,
					'lbl': lbl_file,
				})

	def __len__(self):
		return len(self.files[self.split])

	def __getitem__(self, index):
		data_file = self.files[self.split][index]
		# load image
		img_file = data_file['img']
		img = PIL.Image.open(img_file)
		img = np.array(img, dtype=np.uint8)
		# load label
		lbl_file = data_file['lbl']
		lbl = PIL.Image.open(lbl_file)
		lbl = np.array(lbl, dtype=np.int32)
		lbl[lbl == 255] = -1
		if self._transform:
			return self.transform(img, lbl)
		else:
			return img, lbl

	def transform(self, img, lbl):
		img = img[:, :, ::-1]  # RGB -> BGR
		img = img.astype(np.float64)
		img -= self.mean_bgr
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img).float()
		lbl = torch.from_numpy(lbl).long()
		return img, lbl

	def untransform(self, img, lbl):
		img = img.numpy()
		img = img.transpose(1, 2, 0)
		img += self.mean_bgr
		img = img.astype(np.uint8)
		img = img[:, :, ::-1]
		lbl = lbl.numpy()
		return img, lbl
