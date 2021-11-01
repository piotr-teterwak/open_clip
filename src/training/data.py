import os
import sys
import math
import logging
import functools
import braceexpand
import random
import pdb

import pandas as pd
import numpy as np
import pyarrow as pa
from PIL import Image
from PIL import ImageFilter

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from webdataset.utils import identity
import webdataset as wds

import cv2



from clip.clip import tokenize


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

"""https://github.com/facebookresearch/moco/blob/main/moco/loader.py"""
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def build_simclr_transform():
    augmentation = [
	transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
	transforms.RandomApply([
	    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
	], p=0.8),
	transforms.RandomGrayscale(p=0.2),
	transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	normalize
    ]
    return TwoCropTransform(transforms.Compose(augmentation))


# From https://github.com/Stomper10/CheXpert/blob/62cf19ba92dc316f3f46327be6fc763aa5ae185f/materials.py#L37
class CheXpertDataSet(Dataset):
    def __init__(self, label_path, args,transform_train, mode='train'):
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                args.enchance_index_list.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                args.enchance_index_list.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                #image_path = image_path.replace('/nas/public/CheXpert/CheXpert-v1.0','/research/piotrt/data/chexpert/CheXpert-v1.0-small/')
                image_path = os.path.join('/research/piotrt/data/chexpert',image_path)
                self._image_paths.append(image_path)
                assert os.path.exists(image_path), image_path
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(args.enchance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)
        print(self._num_image)
        self.transform_train = transform_train

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image_orig = Image.fromarray(image).convert('RGB')
        image = self.transform_train(image_orig)
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        return image,labels



class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t",data_prefix=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        if data_prefix is not None:
            self.images = [os.path.join(data_prefix,img_path) for img_path in self.images]
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.data_prefix = data_prefix
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]
        return images, texts

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

def preprocess_txt(text):
    return tokenize([str(text)])[0]

def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes = eval(open(os.path.join(dir_path, 'sizes.json'), 'r').read())
    total_size = sum(
        [int(sizes[os.path.basename(shard)]) for shard in shards_list])
    num_shards = len(shards_list)
    return total_size, num_shards

def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path  = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)


    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=is_train,
        shuffle=shuffle
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_chexpert(args, preprocess_fns, split):
    assert split in ["train", "val"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns


    if is_train:
        data_path  = args.train_data
        preprocess_fn = preprocess_train
    else:
        data_path = args.val_data
        preprocess_fn = preprocess_val
    assert data_path

    dataset = CheXpertDataSet(data_path,args, transform_train=preprocess_fn)


    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True,
        drop_last=is_train,
        shuffle=shuffle
    )
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def get_wds_dataset(args, preprocess_img, is_train):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None

    # The following code is adapted from https://github.com/tmbdev/webdataset-examples/blob/master/main-wds.py
    num_samples, num_shards = get_dataset_size(input_shards)
    if is_train and args.distributed:
        max_shards_per_node = math.ceil(num_shards / args.world_size)
        num_samples = args.world_size * (num_samples * max_shards_per_node // num_shards)
        num_batches = num_samples // (args.batch_size * args.world_size)
        num_samples = num_batches * args.batch_size * args.world_size
    else:
        num_batches = num_samples // args.batch_size
    shardlist = wds.PytorchShardList(
        input_shards,
        epoch_shuffle=is_train,
        split_by_node=is_train  # NOTE: we do eval on a single gpu.
    )
    dataset = (
        wds.WebDataset(shardlist)
        .decode("pil")
        .rename(image="jpg;png", text="txt")
        .map_dict(image=preprocess_img, text=preprocess_txt)
        .to_tuple("image", "text")
        .batched(args.batch_size, partial=not is_train or not args.distributed)
    )
    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers,
    )
    if is_train and args.distributed:
        # With DDP, we need to make sure that all nodes get the same number of batches;
        # we do that by reusing a little bit of data.
        dataloader = dataloader.repeat(2).slice(num_batches)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader, None)

def get_csv_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        data_prefix=args.data_prefix
        )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.dataset_type == "chexpert":
        data["train"] = get_chexpert(args, preprocess_fns, "train")
        data["val"] = get_chexpert(args, preprocess_fns, "val")
    elif args.dataset_type == "imagenet":
        data["train"] = get_imagenet(args, preprocess_fns, "train")
        data["val"] = get_imagenet(args, preprocess_fns, "val")
    else:
      if args.train_data:
          data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
              args, preprocess_train, is_train=True)
      if args.val_data:
          data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
              args, preprocess_val, is_train=False)

    if args.imagenet_zeroshot:
        if args.imagenet_val is not None:
            data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")
        if args.imagenet_v2 is not None:
            data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
