# EVCal_analysis.py

from MuellerLib import *
from tifffile import imwrite, imread, TiffFile
from PIL import Image
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n","--numSamples", type=int, nargs='?', default=1000, help="set number of samples to be compared")
args = parser.parse_args()


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

def ecm_calibrate_N(B, M, rank=4):
    """
    B : (K, N, n, m)  gemessene Matrizen B_k = A M_k W für N Kalibrationen
                      (k=0 muss 'leer' sein)
    M : (K, 4, 4)     bekannte Mueller-Matrizen der Proben (M[0] = Identität)
    Rückgabe:
        A   : (N, n, 4)
        W   : (N, 4, m)
        eps : (N,)       Verhältnis s[-1]/s[-2] je Kalibration
        s   : (N, 4*m)   Singulärwerte von H je Kalibration
    """
    B = np.asarray(B, dtype=float)
    M = np.asarray(M, dtype=float)
    K, N, n, m = B.shape

    # --- Pseudoinverse (Rang 4) von B[0], batched über N -> (N, m, n)
    U0, S0, Vt0 = np.linalg.svd(B[0], full_matrices=False)       # (N,n,r),(N,r),(N,r,m)
    B0PInv = (Vt0[:, :rank, :].transpose(0, 2, 1) / S0[:, None, :rank]) \
             @ U0[:, :, :rank].transpose(0, 2, 1)                 # (N,m,n)

    # --- C_k = W^+ M_k W  -> (K, N, m, m)
    C = B0PInv[None] @ B
    Ct = C.swapaxes(-1, -2)

    # --- H_k = kron(M_k, I_m) - kron(I_4, C_k^T)
    I4, Im = np.eye(4), np.eye(m)
    T1 = np.einsum('kij,ab->kiajb', M, Im).reshape(K, 4*m, 4*m)   # (K,4m,4m)
    T2 = np.einsum('ij,knab->kniajb', I4, Ct).reshape(K, N, 4*m, 4*m)
    H = T1[:, None] - T2                                          # (K,N,4m,4m)
    H = H.transpose(1, 0, 2, 3).reshape(N, K*4*m, 4*m)            # (N,K*4m,4m)

    # --- Nullraum via SVD: letzte Zeile von Vt
    _, s, Vt = np.linalg.svd(H, full_matrices=False)              # s:(N,4m), Vt:(N,4m,4m)
    W = Vt[:, -1, :].reshape(N, 4, m)

    # Skalierung egal (W -> aW, A -> A/a)
    W = W / W[:, 0, :].mean(axis=-1)[:, None, None]

    # --- A per Least Squares: A = C2 C1^-1
    MW = M[:, None] @ W[None]                                     # (K,N,4,m)
    C1 = np.einsum('knij,knlj->nil', MW, MW)                      # (N,4,4)
    C2 = np.einsum('knij,knlj->nil', B, MW)                       # (N,n,4)
    # C1 ist symmetrisch -> A = C2 C1^-1 = (C1^-1 C2^T)^T
    A = np.linalg.solve(C1, C2.transpose(0, 2, 1)).transpose(0, 2, 1)   # (N,n,4)

    eps = s[:, -1] / s[:, -2]
    return A, W, eps, s

calibration_path = "./auswertung/Run1/mueller/calibration/EV"
raw_path = "./auswertung/Run1/mueller/raw/USAV1/path"
Calibration_data = []
for i in range(4):
	Calibration_data.append(load_tile(f"{calibration_path}/Probe{i+1}"))
Calibration_data = np.array(Calibration_data, dtype=np.float32)


a = Calibration_data[:, :,:, :]
l,m,n,o = Calibration_data.shape
evens = np.empty((l, m, int(n/2), o), dtype=a.dtype)
evens[:,0::2,:] = a[:,0::2, 0::2,:]   # gerade Zeilen: Spalten 0, 2, 4, ...
evens[:,1::2,:] = a[:,1::2, 1::2,:]   # ungerade Zeilen: Spalten 1, 3, 5, ...
odds = np.empty((l, m, int(n/2), o), dtype=a.dtype)
odds[:,0::2,:] = a[:,0::2, 1::2,:]  # 01 03 05 ...
odds[:,1::2,:] = a[:,1::2, 0::2,:]  # 10 12 14 ...

np.random.seed(seed = 42)
numberOfSamples = args.numSamples

##################################  EVEN DATA  ####################################

CalData = evens # Calibration_data
m, n = CalData.shape[1:3]

m_rand = np.floor(np.random.rand(numberOfSamples)*m).astype(int)
n_rand = np.floor(np.random.rand(numberOfSamples)*n).astype(int)
random_samples = CalData[:, m_rand, n_rand, :].reshape(4, numberOfSamples, 6, 6)
random_samples.shape

Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
A, W, eps, s = ecm_calibrate_N(random_samples[:,:,:,:], Proben)
A_tst, W_tst, eps_tst, s_tst = ecm_calibrate(random_samples[:,0,:,:], Proben)

# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = A.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])]
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte A Gerade Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_A_even.pdf")
plt.clf()

güte = np.empty((6, 24))
güte[0,:] = std/np.abs(mean)


# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = W.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])]
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte W Gerade Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_W_even.pdf")
plt.clf()
güte[1,:] = std/np.abs(mean)


##################################  ODD DATA  ####################################

CalData = odds # Calibration_data
m, n = CalData.shape[1:3]
m_rand = np.floor(np.random.rand(numberOfSamples)*m).astype(int)
n_rand = np.floor(np.random.rand(numberOfSamples)*n).astype(int)
random_samples = CalData[:, m_rand, n_rand, :].reshape(4, numberOfSamples, 6, 6)
random_samples.shape

Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
A, W, eps, s = ecm_calibrate_N(random_samples[:,:,:,:], Proben)
A_tst, W_tst, eps_tst, s_tst = ecm_calibrate(random_samples[:,0,:,:], Proben)

# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = A.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])]
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte A Ungerade Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_A_odd.pdf")
plt.clf()
güte[2,:] = std/np.abs(mean)


# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = W.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])]
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte W Ungerade Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_W_odd.pdf")
plt.clf()
güte[3,:] = std/np.abs(mean)

##################################  ALL DATA  ####################################

CalData = Calibration_data # Calibration_data
m, n = CalData.shape[1:3]
m_rand = np.floor(np.random.rand(numberOfSamples)*m).astype(int)
n_rand = np.floor(np.random.rand(numberOfSamples)*n).astype(int)
random_samples = CalData[:, m_rand, n_rand, :].reshape(4, numberOfSamples, 6, 6)
random_samples.shape

Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
A, W, eps, s = ecm_calibrate_N(random_samples[:,:,:,:], Proben)
A_tst, W_tst, eps_tst, s_tst = ecm_calibrate(random_samples[:,0,:,:], Proben)

# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = A.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])]
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte A alle Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_A_all.pdf")
plt.clf()
güte[4,:] = std/np.abs(mean)

# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = W.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])]
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte W alle Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_W_all.pdf")
plt.clf()

güte[5,:] = std/np.abs(mean)

meangüte = np.mean(güte, axis=1)
print("std / |mean|")
print("even, odds, all")
print(f"A : {meangüte[::2]} \n B : {meangüte[1::2]}")

# breakpoint()




CalData = evens # Calibration_data
m, n = CalData.shape[1:3]

m_rand = np.floor(np.random.rand(numberOfSamples)*m).astype(int)
n_rand = np.floor(np.random.rand(numberOfSamples)*n).astype(int)
random_samples = CalData[:, m_rand, n_rand, :].reshape(4, numberOfSamples, 6, 6)
random_samples.shape

Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
A, W, eps, s = ecm_calibrate_N(random_samples[:,:,:,:], Proben)
A_tst, W_tst, eps_tst, s_tst = ecm_calibrate(random_samples[:,0,:,:], Proben)

# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = A.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")

plt.figure(figsize=(12, 6))

plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])], label = "evens"
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte A Gerade Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()



CalData = odds # Calibration_data
m, n = CalData.shape[1:3]
m_rand = np.floor(np.random.rand(numberOfSamples)*m).astype(int)
n_rand = np.floor(np.random.rand(numberOfSamples)*n).astype(int)
random_samples = CalData[:, m_rand, n_rand, :].reshape(4, numberOfSamples, 6, 6)
random_samples.shape

Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
A, W, eps, s = ecm_calibrate_N(random_samples[:,:,:,:], Proben)
A_tst, W_tst, eps_tst, s_tst = ecm_calibrate(random_samples[:,0,:,:], Proben)

# Beispiel: N Datensätze mit jeweils 20 Messwerten
data = A.reshape(numberOfSamples,-1)

# Statistische Kennwerte entlang der Zeilen berechnen
mean = np.mean(data, axis=0)
std = np.std(data, axis=0, ddof=1)
median = np.median(data, axis=0)
q1 = np.percentile(data, 25, axis=0)
q3 = np.percentile(data, 75, axis=0)
minimum = np.min(data, axis=0)
maximum = np.max(data, axis=0)
iqr = q3 - q1

# Ergebnisse ausgeben
# for i in range(data.shape[1]):
# 	print(f"\nDatensatz {i+1}:")
# 	print(f"Mean:   {mean[i]:.3}")
# 	print(f"Std:    {std[i]:.3}")
# 	print(f"Median: {median[i]:.3}")
# 	print(f"Q1:     {q1[i]:.3}")
# 	print(f"Q3:     {q3[i]:.3}")
# 	print(f"IQR:    {iqr[i]:.3}")
# 	print(f"Min:    {minimum[i]:.3}")
# 	print(f"Max:    {maximum[i]:.3}")



plt.boxplot(
    data,
    showmeans=True,
    meanline=True,
    patch_artist=True,
    tick_labels=[f"{i+1}" for i in range(data.shape[1])], label = "odds"
)

plt.xlabel("Matrixinhalte")
plt.ylabel("Messwerte")
plt.title("Vergleich der Matrixinhalte A Ungerade Inhalte")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(f"./auswertung/Run1/mueller/calibration/EV/{numberOfSamples}_A_oddAndEven.pdf")
plt.clf()