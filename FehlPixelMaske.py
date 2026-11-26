from MuellerLib import *
from tifffile import imwrite, imread, TiffFile
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import cv2
from pathlib import Path


def getFilesOfType(pathStr:str, Type:str=".csv"):
    DateienOfType = [
        datei for datei in os.listdir(pathStr)
        if datei.lower().endswith(Type)
    ]
    return DateienOfType

def load_tile(tile_path):
    tiffExample = getFilesOfType(f"{tile_path}", ".tiff")
    with TiffFile(f"{tile_path}/{tiffExample[0]}") as tif:
        m, n = np.array(tif.pages[0].asarray()).shape
    tileArray = np.zeros((m, n, 36))
    for i, file in enumerate(getFilesOfType(f"{tile_path}", ".tiff")):
        with TiffFile(f"{tile_path}/{file}") as tif:
            tileArray[:,:,i] = np.array(tif.pages[0].asarray())
    return tileArray


calibration_path = "./auswertung/Run1/mueller/calibration/EV"
raw_path = "./auswertung/Run1/mueller/raw/USAV1/path"
n = 10
for i in range(4):
    Calibration_data = np.array(load_tile(f"{calibration_path}/Probe{i+1}"))
    for j in range(36):
        # Fourier-Transformation
        sorted_arr = np.sort(Calibration_data[:,:,j].ravel())
        Grad = np.gradient(sorted_arr)
        idx = np.where(Grad == np.max(Grad))[0]
        print(Grad[idx], sorted_arr[idx])
        # plt.plot(np.arange(sorted_arr.shape[0]),sorted_arr)
        # plt.plot(np.arange(Grad.shape[0]), Grad)
        # plt.show()
        mask = Calibration_data[:,:,j] >= sorted_arr[idx]
        print(np.sum(mask.ravel()))
        mean_ = np.ma.array(Calibration_data[:,:,j], mask=mask).mean()
        Calibration_data[mask, j] = mean_
        Res_path = Path(f"{calibration_path}/Probe{i+1}_masked")
        Res_path.mkdir(parents=True, exist_ok=True)
        imwrite(f"{calibration_path}/Probe{i+1}_masked/{j}.tiff", Calibration_data[:,:, j].reshape(Calibration_data[:,:,j].shape).astype(np.float32), compression=None)