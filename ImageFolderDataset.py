import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    print("can't find acc image loader")
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MyImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, dir, transform=None):
        image_list = []
        for target in sorted(os.listdir(dir)):
            path = os.path.join(dir, target)
            image_list.append(path)

        self.image_list = image_list
        self.loader = default_loader
        self.transform = transform
        self.len = len(image_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.image_list[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.len


class SingleImg2Tensor(data.Dataset):
    def __init__(self, img_dir, transformer=None):
        super(SingleImg2Tensor, self).__init__()
        self.img_dir = img_dir
        self.transformer = transformer

    def __getitem__(self, index):
        img = pil_loader(self.img_dir[index])
        if self.transformer is not None:
            img = self.transformer(img)
        return img

    def __len__(self):
        # return len(self.img_dir)
        return 1


class MyCoverAndSecretFolder(data.Dataset):
    def __init__(self, cover_dir, secret_dir, transformer=None, scale_size=256):
        cover_image_list = []
        for cover_img_name in sorted(os.listdir(cover_dir)):
            path = os.path.join(cover_dir, cover_img_name)
            cover_image_list.append(path)
        self.cover_image_list = cover_image_list

        secret_image_list = []
        for secret_img_name in sorted(os.listdir(secret_dir)):
            path = os.path.join(secret_dir, secret_img_name)
            secret_image_list.append(path)
        self.secret_image_list = secret_image_list
        self.loader = default_loader
        self.transformer = transformer
        self.len = len(cover_image_list)
        self.resize_scale = scale_size

    def __getitem__(self, index):
        cover_img_path = self.cover_image_list[index]
        cover_img = self.loader(cover_img_path)
        secret_img_path = self.secret_image_list[index]
        secret_img = self.loader(secret_img_path)

        # if self.resize_scale is not None:
        #     secret_img = secret_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
        #     cover_img = cover_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.transformer is not None:
            cover_img = self.transformer(cover_img)
            secret_img = self.transformer(secret_img)

        return cover_img_path, cover_img, secret_img_path, secret_img

    def __len__(self):
        return self.len


class MyKeyCoverAndSecretFolder(data.Dataset):
    def __init__(self, key_dir, cover_dir, secret_dir, transformer=None, scale_size=256):
        key_image_list = []
        for key_img_name in sorted(os.listdir(key_dir)):
            path = os.path.join(key_dir, key_img_name)
            key_image_list.append(path)
        self.key_image_list = key_image_list

        cover_image_list = []
        for cover_img_name in sorted(os.listdir(cover_dir)):
            path = os.path.join(cover_dir, cover_img_name)
            cover_image_list.append(path)
        self.cover_image_list = cover_image_list
        np.random.shuffle(self.cover_image_list)

        secret_image_list = []
        for secret_img_name in sorted(os.listdir(secret_dir)):
            path = os.path.join(secret_dir, secret_img_name)
            secret_image_list.append(path)
        self.secret_image_list = secret_image_list
        np.random.shuffle(self.secret_image_list)

        self.loader = default_loader
        self.transformer = transformer
        self.len = len(cover_image_list)
        self.resize_scale = scale_size

    def __getitem__(self, index):

        key_img_path = self.key_image_list[index]
        key_img = self.loader(key_img_path)
        cover_img_path = self.cover_image_list[index]
        cover_img = self.loader(cover_img_path)
        secret_img_path = self.secret_image_list[index]
        secret_img = self.loader(secret_img_path)

        # if self.resize_scale is not None:
        #     secret_img = secret_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
        #     cover_img = cover_img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.transformer is not None:
            key_img = self.transformer(key_img)
            cover_img = self.transformer(cover_img)
            secret_img = self.transformer(secret_img)

        return key_img_path, key_img, cover_img_path, cover_img, secret_img_path, secret_img

    def __len__(self):
        return self.len
