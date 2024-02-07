import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import skimage.io as sio

from torch.utils.data.dataset import Dataset
from skimage.transform import resize
from tqdm import tqdm

def clip_limit(im, clim=0.01):

    if im.dtype == np.dtype('uint8'):
        hist, *_ = np.histogram(im.reshape(-1),
                                bins=np.linspace(0, 255, 255),
                                density=True)
    elif im.dtype == np.dtype('uint16'):
        hist, *_ = np.histogram(im.reshape(-1),
                                bins=np.linspace(0, 65535, 65536),
                                density=True)
    cumh = 0
    for i, h in enumerate(hist):
        cumh += h
        if cumh > 0.01:
            break
    cumh = 1
    for j, h in reversed(list(enumerate(hist))):
        cumh -= h
        if cumh < (1 - 0.01):
            break
    im = np.clip(im, i, j)
    return im

def normalize(arr):
    arr = clip_limit(arr)
    arr = arr.astype('float32')
    return (arr - arr.min()) / (arr.max() - arr.min())
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.abs(torch.randn(tensor.size()) * self.std + self.mean)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# path relation
class CustomPathLoader(Dataset):
    def __init__(self, image_paths,  target_paths, marker_paths):

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.marker_paths = marker_paths

    def __getitem__(self, index):

        x = self.image_paths[index]
        y = self.target_paths[index]
        z = self.marker_paths[index]

        return x, y, z

    def __len__(self):

        return len(self.image_paths)


class CustomDataLoader(Dataset):
    def __init__(self, df, augment=True, resolution=(512, 512), resize_to=None):

        impaths = list(df.image)
        gtpaths = list(df.label)
        markerpaths = list(df.marker)
        
        if resize_to is not None:
            gts = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([resize(normalize(sio.imread(p)), resize_to, anti_aliasing=True)
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(markerpaths)])
        else:
            gts = np.stack([sio.imread(p)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([normalize(sio.imread(p))
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([sio.imread(p)
                            for p in tqdm(markerpaths)])

        # binarize ground truth for segmentation
        gts = gts > 0
        mks = mks > 0

        self.augment = augment
        self.resolution = resolution
        self.ims = ims.astype('float32')
        self.gts = gts.astype('uint8')
        self.mks = mks.astype('uint8')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        image = self.ims[idx]
        label = self.gts[idx]
        marker = self.mks[idx]

        # Data augmentation
        # -----------------
        if self.augment:

            # ToPILImage
            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)

            # Random horizontal flipping
            if np.random.rand() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                marker = TF.hflip(marker)

            # Random vertical flipping
            if np.random.rand() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                marker = TF.vflip(marker)

            # Random Rotation
            angle = np.random.randint(91)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            marker = TF.rotate(marker, angle)

            # Random crop to 512 x 512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.resolution)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            marker = TF.crop(marker, i, j, h, w)

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)

            # Add random noise
            if np.random.rand() > 0.5:
                noise = np.random.rand() * .2
                image = AddGaussianNoise(std=noise)(image)

        else:
            # ToPILImage
            H, W = image.squeeze().shape
            newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
            newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)

            # # Pad by center cropping with larger dims
            image = TF.center_crop(image, (newH, newW))
            label = TF.center_crop(label, (newH, newW))
            marker = TF.center_crop(marker, (newH, newW))

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
        
        image = torch.cat([image, image, image], dim=0)
        label = torch.cat([label, marker], dim=0)
        return image.type(torch.FloatTensor), (label > 0).type(torch.FloatTensor)

        
class CustomTestDataLoader(Dataset):
    def __init__(self, df):
        self.impaths = list(df.image)

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        image = normalize(sio.imread(self.impaths[idx]))        

        H, W = image.squeeze().shape
        newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
        newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

        # Pad by center cropping with larger dims
        image = TF.to_pil_image(image, mode='F')
        image = TF.center_crop(image, (newH, newW))
        image = TF.to_tensor(image)
        
        image = torch.cat([image, image, image], dim=0)

        return image.type(torch.FloatTensor)
        
        
#####################################################        
# With Motion Cues
#####################################################

# path relation
class CustomMotionPathLoader(Dataset):
    def __init__(self, image_paths, bgs_paths, flux_paths,  target_paths, marker_paths):

        self.image_paths = image_paths
        self.bgs_paths = bgs_paths
        self.flux_paths = flux_paths
        self.target_paths = target_paths
        self.marker_paths = marker_paths

    def __getitem__(self, index):

        img = self.image_paths[index]
        bgs = self.bgs_paths[index]
        flux = self.flux_paths[index]
        mask = self.target_paths[index]
        marker = self.marker_paths[index]

        return img, bgs, flux, mask, marker

    def __len__(self):

        return len(self.image_paths)
        

class CustomMotionTestPathLoader(Dataset):
    def __init__(self, image_paths, bgs_paths, flux_paths):

        self.image_paths = image_paths
        self.bgs_paths = bgs_paths
        self.flux_paths = flux_paths

    def __getitem__(self, index):

        img = self.image_paths[index]
        bgs = self.bgs_paths[index]
        flux = self.flux_paths[index]

        return img, bgs, flux

    def __len__(self):

        return len(self.image_paths)
        

class MUSENetDataLoader(Dataset):
    def __init__(self, df, augment=True, resolution=(512, 512), resize_to=None):

        impaths = list(df.image)
        bgspaths = list(df.bgs)
        fluxpaths = list(df.flux)
        gtpaths = list(df.label)
        markerpaths = list(df.marker)
        
        if resize_to is not None:
            gts = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([resize(normalize(sio.imread(p)), resize_to, anti_aliasing=True)
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(markerpaths)])
            bgs = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(bgspaths)])
            flux = np.stack([resize(sio.imread(p), resize_to, anti_aliasing=False)
                            for p in tqdm(fluxpaths)])
        else:
            gts = np.stack([sio.imread(p)
                            for p in tqdm(gtpaths)], axis=0)
            ims = np.stack([normalize(sio.imread(p))
                            for p in tqdm(impaths)], axis=0)
            mks = np.stack([sio.imread(p)
                            for p in tqdm(markerpaths)])
            bgs = np.stack([sio.imread(p)
                            for p in tqdm(bgspaths)])
            flux = np.stack([sio.imread(p)
                            for p in tqdm(fluxpaths)])

        # binarize ground truth for segmentation
        gts = gts > 0
        mks = mks > 0

        self.augment = augment
        self.resolution = resolution
        self.ims = ims.astype('float32')
        self.bgs = bgs.astype('uint8')
        self.flux = flux.astype('uint8')
        self.gts = gts.astype('uint8')
        self.mks = mks.astype('uint8')
       
    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        image = self.ims[idx]
        bgs = self.bgs[idx]
        flux = self.flux[idx]
        label = self.gts[idx]
        marker = self.mks[idx]

        # Data augmentation
        # -----------------
        if self.augment:

            # ToPILImage
            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            bgs = TF.to_pil_image(bgs)
            flux = TF.to_pil_image(flux)

            # Random horizontal flipping
            if np.random.rand() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                marker = TF.hflip(marker)
                bgs = TF.hflip(bgs)
                flux = TF.hflip(flux)

            # Random vertical flipping
            if np.random.rand() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                marker = TF.vflip(marker)
                bgs = TF.vflip(bgs)
                flux = TF.vflip(flux)

            # Random Rotation
            angle = np.random.randint(91)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            marker = TF.rotate(marker, angle)
            bgs = TF.rotate(bgs, angle)
            flux = TF.rotate(flux, angle)

            # Random crop to 512 x 512
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.resolution)
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)
            marker = TF.crop(marker, i, j, h, w)
            bgs = TF.crop(bgs, i, j, h, w)
            flux = TF.crop(flux, i, j, h, w)

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            bgs = TF.to_tensor(bgs)
            flux = TF.to_tensor(flux)

            # Add random noise
            if np.random.rand() > 0.5:
                noise = np.random.rand() * .2
                image = AddGaussianNoise(std=noise)(image)

        else:
            # ToPILImage
            H, W = image.squeeze().shape
            newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
            newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

            image = TF.to_pil_image(image, mode='F')
            label = TF.to_pil_image(label)
            marker = TF.to_pil_image(marker)
            bgs = TF.to_pil_image(bgs)
            flux = TF.to_pil_image(flux)

            # # Pad by center cropping with larger dims
            image = TF.center_crop(image, (newH, newW))
            label = TF.center_crop(label, (newH, newW))
            marker = TF.center_crop(marker, (newH, newW))
            bgs = TF.center_crop(bgs, (newH, newW))
            flux = TF.center_crop(flux, (newH, newW))

            # Back to tensor
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            marker = TF.to_tensor(marker)
            bgs = TF.to_tensor(bgs)
            flux = TF.to_tensor(flux)
            
        emptyImg = torch.empty((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.uint8)
        
        image = torch.cat([image, image, image], dim=0)
        image2 = torch.cat([bgs, flux, emptyImg], dim=0)
        label = torch.cat([label, marker], dim=0)
        return image.type(torch.FloatTensor), image2.type(torch.FloatTensor), (label > 0).type(torch.FloatTensor)
        

class MUSENetTestDataLoader(Dataset):
    def __init__(self, df):
        self.impaths = list(df.image)
        self.bgspaths = list(df.bgs)
        self.fluxpaths = list(df.flux)

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        image = normalize(sio.imread(self.impaths[idx]))
        bgs = sio.imread(self.bgspaths[idx])
        flux = sio.imread(self.fluxpaths[idx])        

        H, W = image.squeeze().shape
        newH = (H // 32 + 1) * 32 if H % 32 != 0 else H
        newW = (W // 32 + 1) * 32 if W % 32 != 0 else W

        # Pad by center cropping with larger dims
        image = TF.to_pil_image(image, mode='F')
        bgs = TF.to_pil_image(bgs)
        flux = TF.to_pil_image(flux)
            
        image = TF.center_crop(image, (newH, newW))
        bgs = TF.center_crop(bgs, (newH, newW))
        flux = TF.center_crop(flux, (newH, newW))
        
        image = TF.to_tensor(image)
        bgs = TF.to_tensor(bgs)
        flux = TF.to_tensor(flux)
        
        emptyImg = torch.empty((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.uint8)
        
        image = torch.cat([image, image, image], dim=0)
        image2 = torch.cat([bgs, flux, emptyImg], dim=0)
        
        return image.type(torch.FloatTensor), image2.type(torch.FloatTensor)
        