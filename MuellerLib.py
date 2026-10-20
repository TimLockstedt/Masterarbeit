import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from tqdm import tqdm

def MuellerRetarder(delta, theta):
    """
    Returns the Mueller matrix of a retarder with delta as the phase difference and theta as the orientation angle.
    Both in terms of degrees !!!!
    """
    delta_rad = np.deg2rad(delta)
    theta_rad = np.deg2rad(theta)

    c_delta = np.cos(delta_rad)
    s_delta = np.sin(delta_rad)
    c2_theta = np.cos(2 * theta_rad)
    s2_theta = np.sin(2 * theta_rad)

    M = np.array([
        [1, 0, 0, 0],
        [0, c2_theta**2 + s2_theta**2 * c_delta, c2_theta * s2_theta * (1 - c_delta), s2_theta * s_delta],
        [0, c2_theta * s2_theta * (1 - c_delta), s2_theta**2 + c2_theta**2 * c_delta, -c2_theta * s_delta],
        [0, -s2_theta * s_delta, c2_theta * s_delta, c_delta]
    ])
    return M

def MuellerLinearPolarizer(theta):
    """
    Returns the Mueller matrix of a linear polarizer with orientation angle theta.
    In terms of degrees !!!!
    """
    theta_rad = np.deg2rad(theta)
    c2_theta = np.cos(2 * theta_rad)
    s2_theta = np.sin(2 * theta_rad)

    M = 0.5 * np.array([
        [1, c2_theta, s2_theta, 0],
        [c2_theta, c2_theta**2, c2_theta * s2_theta, 0],
        [s2_theta, c2_theta * s2_theta, s2_theta**2, 0],
        [0, 0, 0, 0]
    ])
    return M


def pinv_rank4(B):
    """Pseudoinverse von B0 = A W, abgeschnitten auf Rang 4 (Rauschunterdrückung)."""
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return Vt[:4].T @ np.diag(1/s[:4]) @ U[:, :4].T # Siehe markdown Eintrag

def ecm_calibrate(B, M):
	"""
	B : (K, n, m)  gemessene Matrizen B_k = A M_k W   (k=0 muss 'leer' sein)
	M : (K, 4, 4)  bekannte Mueller-Matrizen der Proben (M[0] = Identität)
	Rückgabe: A (n,4), W (4,m), eps_ECM, Singulärwerte
	"""
	K, n, m = B.shape
	B0PInv = pinv_rank4(B[0])                      # ~ W^+ A^+ -> inverse von (A W)
	# C_k = B0PInv @ B_k  = W^+ A^+ A M_k W = W^+ M_k W
	# -> W C_k = M_k W  <->  M_k W - W C_k = 0 
	# Das Kroneker-Produkt ⊗ wird verwendet, um die Gleichung in eine lineare Form zu bringen, die für alle k gleichzeitig gelöst werden kann
	# Siehe dazu das Markdown Bild unten
	# pro k Vektorisierung: vec(M W - W C) = vec(M W I_m- I_4 W C)
	# = (M ⊗ I_m - I_4 ⊗ CB0^T) vec(W) -> W hat die shape (4,m) -> vec(W) hat die Länge 4*m
	# H W = 0
	H = []
	for k in range(K):
		Ck = B0PInv @ B[k]
		H.append(np.kron(M[k], np.eye(m)) - np.kron(np.eye(4), Ck.T))
	H = np.vstack(H)
	# Gesucht ist H_k vec(X) = 0, mit rauschen ist dies jedoch nun ein minimierungsproblem
	# least squares kann auch über SVD gelöst werden, da die Lösung der kleinste Singulärwert ist
	# Kleinster Singulärwert <-> Eigenvektor von K = H^T H zum kleinsten Eigenwert
	# Die SVD von H = U S V^T -> K = V S U^T U S V^T = V S^2 V^T | V, U orthogonal, S diagonal
	# kleinster eintrag von S^2 ist der kleinste Eigenwert von K, der zugehörige Eigenvektor ist die letzte Spalte von V / Zeile von V^T
	# also reicht es nur die SVD von H zu berechnen und den letzten Eintrag von V^T zu nehmen
	U, s, Vt = np.linalg.svd(H, full_matrices=False)
	# kleinster singulärwert ist der letze diagonaleintrag von s, der zugehörige Eigenvektor ist die letzte Zeile von Vt
	W = Vt[-1].reshape(4, m)

	# Die Skalierung ist hier egal, da aus W -> aW folgt, A -> A/a => B = A/a M aW = A M W
	W = W / W[0].mean()

	# J(A) (- Abweichung vom Messwert) minimieren über alle Proben: B_k ≈ A (M_k W)
	# J(A)  = sum_k || B_k - A (M_k W) ||_F^2 Soll minimiert werden
	#       = sum_k ((B_k - A (M_k W))^T (B_k - A (M_k W)))   
	#   ... = const. + tr(sum_k A (M_k W) (M_k W)^T A^T) - 2 tr(A sum_k B_k (M_k W)^T)   
	#   ... = const. + tr(      A          C1      A^T) - 2 tr(A        C2         )  
	# Ableitung nach A: A C1 A^T    
	# 2 A C1 - 2 C2^T != 0
	# A C1 = C2^T <=> A = C2^T C1^-1 
	C1 = sum((M[k] @ W) @ (M[k] @ W).T for k in range(K))
	C2 = sum(B[k] @ (M[k] @ W).T for k in range(K))
	A = C2 @ np.linalg.inv(C1)

	eps_ecm = s[-1] / s[-2]                     # verhältnis der kleinsten Singulärwerte, sollte sehr klein sein, s[-1] -> 0
	return A, W, eps_ecm, s

def reconstMueller(A:np.ndarray, Res:np.ndarray, W:np.ndarray):
	"""
	Args:
		A : Generator 
		Res : Ergebisse
		W : Analysator

	Returns:
		np.ndarray: Rekonstruierte Muellermatritzen
	"""
	Designmatrix = np.kron(A, W.T)   
	Inverse = np.linalg.pinv(Designmatrix)
	reconst = Res @ Inverse.T
	return reconst.reshape(-1, 4, 4)

def LuChipman(M, eps=1e-12):
    """
    M: (N,4,4) Mueller-Matrizen
    Rückgabe (alle Winkel in rad, shape (N,)):
      diattenuation, depolarization, linear_ret, circular_ret, orientation
    Zerlegung: M = M_Delta @ M_R @ M_D
    """
    M = np.asarray(M, dtype=float)
    M = M / M[:, :1, :1]                      # Normierung auf M00 = 1

    # ---- 1) Diattenuator M_D -------------------------------------------
    Dvec = M[:, 0, 1:]                        # (N,3)
    D = np.linalg.norm(Dvec, axis=1)          # Diattenuation
    D = np.minimum(D, 1 - eps)                # D=1 wäre singulär
    Dhat = Dvec / np.maximum(np.linalg.norm(Dvec, axis=1), eps)[:, None]
    Dvec = Dhat * D[:, None]

    sq = np.sqrt(1 - D**2)
    I3 = np.eye(3)
    mD = sq[:, None, None] * I3 + (1 - sq)[:, None, None] * (Dhat[:, :, None] * Dhat[:, None, :])

    MD = np.zeros_like(M)
    MD[:, 0, 0] = 1
    MD[:, 0, 1:] = Dvec
    MD[:, 1:, 0] = Dvec
    MD[:, 1:, 1:] = mD

    # ---- 2) M' = M_Delta @ M_R = M @ M_D^-1 ----------------------------
    Mp = M @ np.linalg.inv(MD)
    mp = Mp[:, 1:, 1:]                        # (N,3,3)

    # ---- 3) Depolarisator m_Delta über Eigenwerte von m' m'^T ----------
    A = mp @ mp.transpose(0, 2, 1)
    lam = np.clip(np.linalg.eigvalsh(A), 0, None)   # (N,3)
    s = np.sqrt(lam)
    s1 = s.sum(1)
    s2 = s[:, 0]*s[:, 1] + s[:, 1]*s[:, 2] + s[:, 2]*s[:, 0]
    s3 = s.prod(1)                            # = |det m'|
    sign = np.where(np.linalg.det(mp) < 0, -1.0, 1.0)

    lhs = A + s2[:, None, None] * I3
    rhs = s1[:, None, None] * A + s3[:, None, None] * I3
    mDelta = sign[:, None, None] * np.linalg.solve(lhs, rhs)

    depol = 1 - np.abs(np.trace(mDelta, axis1=1, axis2=2)) / 3

    # ---- 4) Retarder m_R = m_Delta^-1 @ m' -----------------------------
    mR = np.linalg.solve(mDelta, mp)

    tr = np.trace(mR, axis1=1, axis2=2)
    R = np.arccos(np.clip((tr - 1) / 2, -1, 1))     # Gesamtretardance
    sinR = np.sin(R)
    f = np.where(sinR > 1e-9, R / (2 * np.where(sinR > 1e-9, sinR, 1)), 0.5)

    a1 = f * (mR[:, 1, 2] - mR[:, 2, 1])      # Q-Komponente
    a2 = f * (mR[:, 2, 0] - mR[:, 0, 2])      # U-Komponente
    a3 = f * (mR[:, 0, 1] - mR[:, 1, 0])      # V-Komponente

    lin_ret = np.hypot(a1, a2)
    circ_ret = a3
    orientation = 0.5 * np.arctan2(a2, a1)    # schnelle Achse, in (-π/2, π/2]

    return D, depol, lin_ret, circ_ret, orientation

def lin_ret_orientation_fast(M, eps=1e-12):
    m = M[:, 1:, 1:] / M[:, :1, :1]
    v1 = m[:, 1, 2] - m[:, 2, 1]
    v2 = m[:, 2, 0] - m[:, 0, 2]
    v3 = m[:, 0, 1] - m[:, 1, 0]
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    vn = np.sqrt(v1*v1 + v2*v2 + v3*v3)
    rho = np.hypot(tr, vn)
    R = np.arctan2(vn, tr) + np.arcsin(np.clip(vn / (2*np.maximum(rho, eps)), -1, 1))
    R = np.minimum(R, np.pi)

    lin = R * np.hypot(v1, v2) / np.maximum(vn, eps)
    theta = 0.5 * np.arctan2(v2, v1)
    return lin, theta



def lu_chipman_svd(M, eps=1e-12):
    """
    M: (N,4,4) Mueller-Matrizen
    Rückgabe (Winkel in rad, je shape (N,)):
      diattenuation, depolarization, linear_ret, circular_ret, orientation
    Zerlegung: M = M_Delta @ M_R @ M_D
    """
    M = np.asarray(M, dtype=float)
    N = M.shape[0]
    M = M / M[:, :1, :1]
    I3 = np.eye(3)

    # ---- 1) Diattenuator -------------------------------------------------
    Dvec = M[:, 0, 1:]
    Dn = np.linalg.norm(Dvec, axis=1)
    D = np.minimum(Dn, 1 - eps)
    Dhat = Dvec / np.maximum(Dn, eps)[:, None]
    Dvec = Dhat * D[:, None]

    sq = np.sqrt(1 - D**2)
    mD = (sq[:, None, None] * I3
          + (1 - sq)[:, None, None] * (Dhat[:, :, None] * Dhat[:, None, :]))
    MD = np.zeros_like(M)
    MD[:, 0, 0] = 1
    MD[:, 0, 1:] = Dvec
    MD[:, 1:, 0] = Dvec
    MD[:, 1:, 1:] = mD

    # ---- 2) Diattenuation herausrechnen: M' = M @ M_D^-1 ------------------
    mp = (M @ np.linalg.inv(MD))[:, 1:, 1:]

    # ---- 3) SVD: m' = U diag(S) Vt  ---------------------------------------
    U, S, Vt = np.linalg.svd(mp)
    Q = U @ Vt                                   # orthogonal, det = ±1
    s = np.where(np.linalg.det(Q) < 0, -1.0, 1.0)

    depol = 1 - S.sum(axis=1) / 3                # Δ = 1 - |tr(mΔ)|/3
    mR = s[:, None, None] * Q                    # Rotation, det = +1

    # ---- 4) Retardance-Vektor aus mR --------------------------------------
    v = np.stack([mR[:, 1, 2] - mR[:, 2, 1],
                  mR[:, 2, 0] - mR[:, 0, 2],
                  mR[:, 0, 1] - mR[:, 1, 0]], axis=1)          # = 2 sin(R) * n
    vn = np.linalg.norm(v, axis=1)
    cosR = (np.trace(mR, axis1=1, axis2=2) - 1) / 2
    R = np.arctan2(vn / 2, cosR)                 # stabiler als arccos

    # Achse: normal aus v, nahe R = π aus dem symmetrischen Anteil
    n = v / np.maximum(vn, eps)[:, None]

    near_pi = (vn < 1e-3) & (cosR < 0)
    if near_pi.any():
        idx = np.nonzero(near_pi)[0]
        Ssym = 0.5 * (mR[idx] + mR[idx].transpose(0, 2, 1))
        B = (Ssym - cosR[idx, None, None] * I3) / (1 - cosR[idx])[:, None, None]
        k = np.argmax(np.einsum('nii->ni', B), axis=1)
        col = B[np.arange(len(idx)), :, k]
        col /= np.maximum(np.linalg.norm(col, axis=1), eps)[:, None]
        sg = np.sign(np.einsum('ni,ni->n', col, v[idx]))
        sg[sg == 0] = 1.0
        n[idx] = col * sg[:, None]

    a = R[:, None] * n                           # Retardance-Vektor
    lin_ret = np.hypot(a[:, 0], a[:, 1])
    circ_ret = a[:, 2]
    orientation = 0.5 * np.arctan2(a[:, 1], a[:, 0])

    return D, depol, lin_ret, circ_ret, orientation