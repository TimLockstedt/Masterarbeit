from MuellerLib import *

name = "20261112"
# Ordner erstellen
# createArchitecture(name)
# Kurven Fitten
Calibration_Data = Datensatz(name)
# Calibration_Data.fitLCR()
# Spannungen für gewollte retardierungen berechnen und abspeichern
Ret = np.array([np.deg2rad(90), np.deg2rad(180), np.deg2rad(270), np.deg2rad(360)])
Calibration_Data.getVoltageMap(Ret)
# # EV Calibration durchführen
# # Optional die idealen Mueller Matrizen der Proben angeben, ansonsten werden die Standardwerte verwendet
# # Proben = np.array([np.eye(4), MuellerLinearPolarizer(0), MuellerLinearPolarizer(45), MuellerRetarder(90,0)])
# Calibration_Data.ecm_calibrate() # (Proben)