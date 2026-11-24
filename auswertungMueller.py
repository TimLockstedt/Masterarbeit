from MuellerLib import *
from tifffile import imwrite, imread, TiffFile
from tqdm import tqdm
import argparse
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

ROOT = Path(__file__).resolve()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--measurement-inputs", type=Path, required=True, help="path to the measurement inputs")
parser.add_argument("-s", "--session-name", type=Path, required=True, help="path to the session")
parser.add_argument("--scraps", type=int, nargs='?', default=False, help="create scrap images")
parser.add_argument("--calibrated", type=bool, nargs='?', default=False, help="if calibration Matrix A and W already exist set to True")
args = parser.parse_args()

directory = f"auswertung/{args.session_name}/mueller/raw"
full_path = f"./{directory}/{args.measurement_inputs}"
tileOrdner = [p for p in Path(full_path).iterdir() if p.is_dir()]

# Bilder Zerlegen 
if args.scraps != False:
    for tile in tqdm(tileOrdner):
        scrap_path = Path(f"{directory}/{args.measurement_inputs}_scraps/{tile.name}")
        scrap_path.mkdir(parents=True, exist_ok=True)
        for file in getFilesOfType(f"{tile}", ".tiff"):
            with TiffFile(f"{tile}/{file}") as tif:
                imgarray = np.array(tif.pages[0].asarray())[:args.scraps, :args.scraps]
                imwrite(f"{directory}/{args.measurement_inputs}_scraps/{tile.name}/{file}", imgarray) 
    print("scraps created! \n" \
    f"find them under {directory}/{args.measurement_inputs}_scraps")
    exit()

# Calibrationsskript
calibrationPath = f"./auswertung/{args.session_name}/mueller/calibration/EV/"
if not args.calibrated:
    calibrationDirs = [p for p in Path(calibrationPath).iterdir() if p.is_dir()]
    calibrationData = np.zeros((4,36))
    print(f"Calibrations ordner Reihenfolge: {calibrationDirs}")
    for i in range(4):
        temp = load_tile(f"{calibrationDirs[i]}")
        temp[temp > 256] = 256 # MUSS ANGEPASST WERDEN FÜR SINNVOLLE MAXIMALWERTE !!!
        calibrationData[i, :] = np.mean(temp.reshape(-1, 36), axis=0)
    calibrationData = calibrationData.reshape(4, 6, 6)        
    Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
    A, W, eps, s = ecm_calibrate(calibrationData, Proben)
    np.savetxt(f"{calibrationPath}/calibrationMatrixA.txt", A)
    np.savetxt(f"{calibrationPath}/calibrationMatrixW.txt", W)
else:
    A = np.loadtxt(f"{calibrationPath}/calibrationMatrixA.txt")
    W = np.loadtxt(f"{calibrationPath}/calibrationMatrixW.txt")

# Auswertungsskript
# ERINNERUNG
# directory = f"auswertung/{args.session_name}/mueller/raw"
# full_path = f"./{directory}/{args.measurement_inputs}"
# tileOrdner = [p for p in Path(full_path).iterdir() if p.is_dir()]
ergebnis_dir = f"auswertung/{args.session_name}/mueller/ergebnisse/{args.measurement_inputs}"
for directory in tileOrdner:
    SRes = load_tile(directory)
    m, n, o = SRes.shape
    MuellerReconst = reconstMueller(A, SRes, W)
    try:
        retardierung, orientierung, diattenuation, diattenAngle = differentialDecompositionFull(MuellerReconst)
        retardierung = retardierung.reshape((m, n))
        orientierung = orientierung.reshape((m, n))
        diattenuation = diattenuation.reshape((m, n)) 
        diattenAngle = diattenAngle.reshape((m, n))
        Res_path = Path(f"{ergebnis_dir}/{directory.name}_differentiell")
        Res_path.mkdir(parents=True, exist_ok=True)
        FilePath = f"{ergebnis_dir}/{directory.name}_differentiell/{args.session_name}_{directory.name}"
        imwrite(f"{FilePath}_retardierung.tiff", retardierung.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}_orientierung.tiff", orientierung.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}_diattenuation.tiff", diattenuation.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}_diattenAngle.tiff", diattenAngle.astype(np.float32), compression=None) 
    except Exception as e :
        print(f"Differentielle zerlegung nicht geglückt: {e}")
    try:
        diatenuation, depolarisation, retardierung, circ_Ret, orientierung,  = lu_chipman_svd(MuellerReconst) # D, depol, lin_ret, circ_ret, orientation
        retardierung = retardierung.reshape((m, n))
        orientierung = orientierung.reshape((m, n))
        diatenuation = diatenuation.reshape((m, n)) 
        depolarisation = depolarisation.reshape((m, n))
        circ_Ret = circ_Ret.reshape((m, n))
        Res_path = Path(f"{ergebnis_dir}/{directory.name}_polar")
        Res_path.mkdir(parents=True, exist_ok=True)
        FilePath = f"{ergebnis_dir}/{directory.name}_polar/{args.session_name}_{directory.name}"
        imwrite(f"{FilePath}_retardierung.tiff", retardierung.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}_orientierung.tiff", orientierung.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}_diatenuation.tiff", diatenuation.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}_depolarisation.tiff", depolarisation.astype(np.float32), compression=None) 
        imwrite(f"{FilePath}circRet.tiff", circ_Ret.astype(np.float32), compression=None) 
    except Exception as e :
        print(f"Polare zerlegung nicht geglückt: {e}")
    RecImage = MuellerReconst.reshape(m, n, 4, 4)
    RecImage = RecImage.transpose(2, 0, 3, 1)
    RecImage = RecImage.reshape(4*m, 4*n)
    Res_path = Path(f"{ergebnis_dir}")
    Res_path.mkdir(parents=True, exist_ok=True)
    imwrite(f"{ergebnis_dir}/{directory.name}_MuellerMatrix.tiff", RecImage.astype(np.float32), compression=None)




# Uncalibrated Matrcies, theoretical method
# retardences = np.array([[2, 1, 0.5, 0.5, 0.5, 0.5], [2, 2, 0.5, 1.5, 2, 1]])*np.pi




# tiffExample = getFilesOfType(f"{tileOrdner[0]}", ".tiff")
# with TiffFile(f"{tileOrdner[0]}/{tiffExample[0]}") as tif:
#     m, n = np.array(tif.pages[0].asarray()).shape
# for tile in tqdm(tileOrdner):
#     cmw = np.zeros((m, n, 36))
#     for i, file in enumerate(getFilesOfType(f"{tile}", ".tiff")):
#         with TiffFile(f"{tile}/{file}") as tif:
#             cmw[:,:,i] = np.array(tif.pages[0].asarray())
    