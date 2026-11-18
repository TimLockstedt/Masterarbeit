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

ROOT = Path(__file__).resolve()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--measurement-inputs", type=Path, required=True, help="path to the measurement inputs")
parser.add_argument("-s", "--session-name", type=Path, required=True, help="path to the session")
parser.add_argument("-c", "--center", type=int, nargs=2, default=(133, 133), metavar=("X", "Y"),
                    help="camera centre on the panel (panel_camera_center.py)")
parser.add_argument("-o", "--output", type=Path, required=False, help="output folder")
parser.add_argument("--instrument", type=Path, default=ROOT / "config" / "sp01")
parser.add_argument("-r", "--radius", type=int, default=15)
parser.add_argument("-e", "--exposure-ms", type=float, default=500)
parser.add_argument("-v", "--voltage-range", type=float, nargs=2, default=(0.0, 5.0), metavar=("MIN", "MAX"))
parser.add_argument("-n", "--steps", type=int, default=101)
parser.add_argument("--settle-ms", type=float, help="override the LCVR settle time")
parser.add_argument("--no-window", action="store_true", help="no window")
parser.add_argument("--scraps", type=int, nargs='?', default=False, help="create scrap images")
parser.add_argument("--calibrated", type=bool, nargs='?', default=False, help="if calibration Matrix A and W already exist set to True")
args = parser.parse_args()

directory = f"auswertung/{args.session_name}/mueller/raw"
full_path = f"./{directory}/{args.measurement_inputs}"
tileOrdner = [p for p in Path(full_path).iterdir() if p.is_dir()]

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

def load_tile(tile_path):
    tiffExample = getFilesOfType(f"{tile_path}", ".tiff")
    with TiffFile(f"{tile_path}/{tiffExample[0]}") as tif:
        m, n = np.array(tif.pages[0].asarray()).shape
    tileArray = np.zeros((m, n, 36))
    for i, file in enumerate(getFilesOfType(f"{tile_path}", ".tiff")):
        with TiffFile(f"{tile_path}/{file}") as tif:
            tileArray[:,:,i] = np.array(tif.pages[0].asarray())
    return tileArray


calibrationPath = f"./auswertung/{args.session_name}/mueller/calibration/EV/"
if not args.calibrated:
    calibrationDirs = [p for p in Path(calibrationPath).iterdir() if p.is_dir()]
    calibrationData = np.zeros((4,36))
    for i in range(4):
        temp = load_tile(f"{calibrationDirs[i]}")
        temp[temp > 256] = 256 # MUSS ANGEPASST WERDEN FÜR SINNVOLLE MAXIMALWERTE !!!
        calibrationData[i, :] = np.mean(temp.reshape(-1, 36), axis=0)
    # probe2 = load_tile(calibrationDirs[1])
    # probe3 = load_tile(calibrationDirs[2])
    # probe4 = load_tile(calibrationDirs[3])
    # histData = probe1[:,:,0].ravel()
    # histData[histData > 50] = 50
    # plt.hist(histData, bins=10)
    calibrationData = calibrationData.reshape(4, 6, 6)        
    Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])            # Zeile n = Analysator, Spalte m = Generator
    A, W, eps, s = ecm_calibrate(calibrationData, Proben)
    np.savetxt(f"{calibrationPath}/calibrationMatrixA.txt", A)
    np.savetxt(f"{calibrationPath}/calibrationMatrixW.txt", W)
else:
    A = np.loadtxt(f"{calibrationPath}/calibrationMatrixA.txt")
    W = np.loadtxt(f"{calibrationPath}/calibrationMatrixW.txt")


# ERINNERUNG
# directory = f"auswertung/{args.session_name}/mueller/raw"
# full_path = f"./{directory}/{args.measurement_inputs}"
# tileOrdner = [p for p in Path(full_path).iterdir() if p.is_dir()]
ergebnis_dir = f"auswertung/{args.session_name}/mueller/ergebnisse/{args.measurement_inputs}"
for directory in tileOrdner:
    SRes = load_tile(directory)
    m, n, o = SRes.shape
    MuellerReconst = reconstMueller(A, SRes, W)
    retardierung, orientierung, C1, C2 = differentialDecompositionFull(MuellerReconst)
    retardierung = retardierung.reshape((m, n))
    orientierung = orientierung.reshape((m, n))
    C1 = C1.reshape((m, n)) 
    C2 = C2.reshape((m, n))
    Res_path = Path(f"{ergebnis_dir}/{directory.name}_res")
    Res_path.mkdir(parents=True, exist_ok=True)
    imwrite(f"{ergebnis_dir}/{directory.name}_res/{args.session_name}_{directory.name}_retardierung.tiff", retardierung.astype(np.float32)
, compression=None) 
    imwrite(f"{ergebnis_dir}/{directory.name}_res/{args.session_name}_{directory.name}_orientierung.tiff", orientierung.astype(np.float32)
, compression=None) 
    imwrite(f"{ergebnis_dir}/{directory.name}_res/{args.session_name}_{directory.name}_C1.tiff", C1.astype(np.float32)
, compression=None) 
    imwrite(f"{ergebnis_dir}/{directory.name}_res/{args.session_name}_{directory.name}_C2.tiff", C2.astype(np.float32)
, compression=None) 

# tiffExample = getFilesOfType(f"{tileOrdner[0]}", ".tiff")
# with TiffFile(f"{tileOrdner[0]}/{tiffExample[0]}") as tif:
#     m, n = np.array(tif.pages[0].asarray()).shape
# for tile in tqdm(tileOrdner):
#     cmw = np.zeros((m, n, 36))
#     for i, file in enumerate(getFilesOfType(f"{tile}", ".tiff")):
#         with TiffFile(f"{tile}/{file}") as tif:
#             cmw[:,:,i] = np.array(tif.pages[0].asarray())
    