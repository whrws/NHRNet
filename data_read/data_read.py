from os import listdir
from os.path import join
from PIL import Image, ImageFile
import torch.utils.data as data
import os
from random import randrange
import cv2
from tqdm import tqdm
import torchvision
from torchvision import transforms
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_file(filename):
  filename_lower = filename.lower()
  return any(filename_lower.endswith(extension) for extension in ['.png', '.jpg', '.bmp', '.mat'])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.data_dir = "{}/hazy/".format(image_dir)
        self.label_dir = "{}/clear/".format(image_dir)

        self.data_filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if is_image_file(x)]
        self.label_filenames = [join(self.label_dir, x) for x in listdir(self.label_dir) if is_image_file(x)]
        self.data_file_num = len(self.data_filenames)
        self.label_file_num = len(self.label_filenames)
        self.transform = transform

        self.crop_width = 224
        self.crop_height = 224

    def __getitem__(self, index):

        data_path_name = self.data_filenames[index]
        data = Image.open(data_path_name)
        width, height = data.size

        data_name = data_path_name.split('/')[-1]
        data_name = data_name.split('_')[0] + '_' + data_name.split('_')[1] + '.png'

        label_img = data_name.split('_')[0] + '.png'
        label_path_name = os.path.join(self.label_dir, label_img)
        label = Image.open(label_path_name)

        if label.mode != 'RGB':
            label = label.convert('RGB')

        if width < self.crop_width or height < self.crop_height:
            raise Exception('Bad image size: {}'.format(label_img))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - self.crop_width + 1), randrange(0, height - self.crop_height + 1)
        haze_crop_img = data.crop((x, y, x + self.crop_width, y + self.crop_height))
        gt_crop_img = label.crop((x, y, x + self.crop_width, y + self.crop_height))

        if self.transform:
            data = self.transform(haze_crop_img)
            label = self.transform(gt_crop_img)

        return data, label

    def __len__(self):
        return len(self.data_filenames)

class TestFromFolder(data.Dataset):
    def __init__(self, image_dir, transform=None):
        super(TestFromFolder, self).__init__()
        self.data_dir = "{}/hazy/".format(image_dir)
        self.label_dir = "{}/gt/".format(image_dir)

        self.data_filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if is_image_file(x)]
        self.label_filenames = [join(self.label_dir, x) for x in listdir(self.label_dir) if is_image_file(x)]
        self.data_file_num = len(self.data_filenames)
        self.label_file_num = len(self.label_filenames)
        self.transform = transform

    def __getitem__(self, index):
        data_path_name = self.data_filenames[index]
        data = Image.open(data_path_name)

        data_name = data_path_name.split('/')[-1]
        label_img = data_name.split('_')[0] + '.png'
        label_path_name = os.path.join(self.label_dir, label_img)
        label = Image.open(label_path_name)

        if label.mode != 'RGB':
            label = label.convert('RGB')

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)
    
    
def crop(path, save_path, crop_size):
    blur_img = [os.path.join(path, imgname) for imgname in sorted(os.listdir(path))]
    save_blur = save_path

    if not os.path.exists(save_blur):
        os.makedirs(save_blur)

    num = 0

    for img in tqdm(blur_img):
        blur_img = cv2.imread(img)
        key_blur = os.path.basename(img).split('.png')[0]
        blur_h, blur_w, blur_c = blur_img.shape
        cnt = 0
        i = 0
        num = num + 1

        while i + crop_size <= blur_h:
            j = 0
            while j + crop_size <= blur_w:
                cnt += 1
                crop_blur_img = blur_img[i:i + crop_size, j:j + crop_size, :]
                crop_blur_path = os.path.join(save_blur, key_blur + '_' + str(cnt) + '.png')
                cv2.imwrite(crop_blur_path, crop_blur_img)
                j += 64
            i = i + 64

class NHDatasetFromFolder(data.Dataset):
      def __init__(self, image_dir, transform=None):
          super(NHDatasetFromFolder, self).__init__()
          self.data_dir_path = "{}/hazy/".format(image_dir)
          self.label_dir_path = "{}/gt/".format(image_dir)
          self.data_dir = "./dataset/NH/crop_hazy/"
          self.label_dir = "./dataset/NH/crop_gt/"
          if not os.path.exists(self.data_dir):
              os.makedirs(self.data_dir)
          if not os.path.exists(self.label_dir):
              os.makedirs(self.label_dir)
          crop_size = 224
          crop(self.data_dir_path, self.data_dir, crop_size)
          crop(self.label_dir_path, self.label_dir, crop_size)
          self.data_filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if is_image_file(x)]
          self.label_filenames = [join(self.label_dir, x) for x in listdir(self.label_dir) if is_image_file(x)]
          self.data_file_num = len(self.data_filenames)
          self.label_file_num = len(self.label_filenames)
          self.transform = transform

      def __getitem__(self, index):

          data_path_name = self.data_filenames[index]
          data = Image.open(data_path_name)

          data_name = data_path_name.split('/')[-1]

          label_img = data_name.split('hazy')[0] + 'GT' + data_name.split('hazy')[1]
          label_path_name = os.path.join(self.label_dir, label_img)
          label = Image.open(label_path_name)

          if label.mode != 'RGB':
              label = label.convert('RGB')

          haze_crop_img, gt_crop_img = self.augment(data, label)

          if self.transform:
              data = self.transform(haze_crop_img)
              label = self.transform(gt_crop_img)

          return data, label

      def augment(self, hazy, clean):
          augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
          rotate_degree = random.choice([90, 180, 270])
          '''Rotate'''
          if augmentation_method == 0:
              hazy = transforms.functional.rotate(hazy, rotate_degree)
              clean = transforms.functional.rotate(clean, rotate_degree)
              return hazy, clean
          '''Vertical'''
          if augmentation_method == 1:
              vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
              hazy = vertical_flip(hazy)
              clean = vertical_flip(clean)
              return hazy, clean
          '''Horizontal'''
          if augmentation_method == 2:
              horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
              hazy = horizontal_flip(hazy)
              clean = horizontal_flip(clean)
              return hazy, clean
          '''no change'''
          if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
              return hazy, clean

      def __len__(self):
          return len(self.data_filenames)


class NHTestFromFolder(data.Dataset):
    def __init__(self, image_dir, transform = None):
        super(NHTestFromFolder, self).__init__()

        self.data_dir = "{}/hazy/".format(image_dir)
        self.label_dir = "{}/gt/".format(image_dir)

        self.data_filenames = [join(self.data_dir, x) for x in listdir(self.data_dir) if is_image_file(x)]
        self.label_filenames = [join(self.label_dir, x) for x in listdir(self.label_dir) if is_image_file(x)]
        self.data_file_num = len(self.data_filenames)
        self.label_file_num = len(self.label_filenames)
        self.transform = transform

    def __getitem__(self, index):
        data_path_name = self.data_filenames[index]
        data = Image.open(data_path_name)

        data_name = data_path_name.split('/')[-1]

        label_img = data_name.split('hazy')[0] + 'GT' + data_name.split('hazy')[1]
        label_path_name = os.path.join(self.label_dir, label_img)
        label = Image.open(label_path_name)

        if label.mode != 'RGB':
            label = label.convert('RGB')

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        return data, label

    def __len__(self):
        return len(self.data_filenames)
    
    

