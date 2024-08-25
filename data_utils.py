from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import xarray as xr
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class CMIP6Dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.models = ["access_cm2", "bcc_csm2_mr", "canesm5_canoe", "cnrm_cm6_1_hr", "miroc6", "miroc_es2l", "mri_esm2_0", 
        "noresm2_mm", "taiesm1", "cesm2", "cmcc_cm2_sr5", "cnrm_esm2_1", "ec_earth3_veg_lr", "kace_1_0_g", "mpi_esm1_2_lr", 
        "noresm2_lm", "ukesm1_0_ll"]
        self.cmip6_root = "/data/home/scv7343/run/SRGAN-master/data/cmip6"


class ERA5LandDataset(CMIP6Dataset):
    def __init__(self, flag) -> None:
        super().__init__()
        self.era5_land_path = "/data/home/scv7343/run/SRGAN-master/data/data.nc"
        assert flag in ["train", "val", "test"]
        self.slice = self._get_slice(flag)
        self._read_era5_land()
        self._read_cmip6()
        
    def _get_slice(self, flag):
        time_dict = {
            "train": slice("1950", "1999"), 
            "val": slice("2000", "2007"), 
            "test": slice("2008", "2014")
        }
        return time_dict[flag]
    
    def _read_era5_land(self):
        ds = xr.open_dataset(self.era5_land_path)
        ds = ds.sel(time=self.slice, expver=1)
        self.era5_land = ds
        self.lon = self.era5_land.longitude.values
        self.lat = self.era5_land.latitude.values
        self.era5_land = self.era5_land.sel(time=self.slice, longitude=self.lon, latitude=self.lat)
        self.era5_land = np.stack([self.era5_land.swvl1.values, self.era5_land.swvl2.values, self.era5_land.swvl3.values], axis=1) # T, C, H, W

    def _read_cmip6(self):
        lst = [join(self.cmip6_root, f"{x}_historical.nc") for x in self.models]
        data = []
        for x in lst:
            ds = xr.open_dataset(x)
            ds = ds.sel(time=self.slice, lon=self.lon, lat=self.lat)
            data.append(ds.mrsos.values / 100)
        self.cmip6 = np.stack(data, axis=1)

    def __getitem__(self, index):
        return self.cmip6[index, ...], self.era5_land[index, ...]

    def __len__(self):
        return len(self.era5_land)
    

class AMSMQTPDataset(CMIP6Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.slice = slice("2000", "2014")
        self._read_amsmqtp()
        self._read_ismn()
        self._read_cmip6()
        self._read_bound()
        
    def _read_amsmqtp(self):
        ds = xr.open_dataset("/data/home/scv7343/run/SRGAN-master/data/AMSMQTP_ensemble_monthly_concat.nc")
        self.lon = ds.lon.values
        self.lat = ds.lat.values
        self.amsmqtp = ds.sel(time=self.slice, layer=range(1, 6), lon=self.lon, lat=self.lat).sm.values # T, C, H, W

    def _read_ismn(self):
        ds_reg = xr.open_dataset("/data/home/scv7343/run/SRGAN-master/data/ismn_monthly_data_rf_reg.nc")
        ds_class = xr.open_dataset("/data/home/scv7343/run/SRGAN-master/data/ismn_monthly_data_rf_class.nc")
        self.ismn_sm = ds_reg.sel(time=self.slice, layer=range(1, 6), lon=self.lon, lat=self.lat).soil_moisture.values
        self.ismn_prob = ds_class.sel(time=self.slice, layer=range(1, 6), lon=self.lon, lat=self.lat).prob.values

    def _read_bound(self):
        ds = xr.open_dataset("/data/home/scv7343/run/SRGAN-master/data/boundary_0.1.nc")
        self.bound = ds.sel(lon=self.lon, lat=self.lat).Band1.values
        self.bound = np.nan_to_num(self.bound)

    def _read_cmip6(self):
        lst = [join(self.cmip6_root, f"{x}_historical.nc") for x in self.models]
        data = []
        for x in lst:
            ds = xr.open_dataset(x)
            ds = ds.sel(time=self.slice, lon=self.lon, lat=self.lat)
            data.append(ds.mrsos.values / 100)
        self.cmip6 = np.stack(data, axis=1)

    def __getitem__(self, index):
        return self.cmip6[index, ...], self.amsmqtp[index, ...], self.ismn_sm[index, ...], self.ismn_prob[index, ...], self.bound

    def __len__(self):
        return len(self.amsmqtp)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
