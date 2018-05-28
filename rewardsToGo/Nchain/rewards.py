import numpy as np
import matplotlib.pyplot as plt
PVFfiles=["PVF_WE_MV", "PVF_WE_QS", "PVF_QS_MV","PVF_QS_QS", "QL", "PVF_QSS_MV", "PVF_QSS_QS"]
GBQLfiles=["GBQL_WE_QS", "GBQL_WE_MV", "GBQL_MU_QS", "GBQL_MU_MV", "GBQL_WE_UC", "GBQL_MU_UC", "QL"]

plt.figure(200)
for file in PVFfiles:
    d=np.genfromtxt(file, delimiter=',')
    plt.plot(range(len(d)), d, label=file)
plt.legend(loc='lower left')
plt.show()
plt.figure(300)
for file in GBQLfiles:
    d=np.genfromtxt(file, delimiter=',')
    plt.plot(range(len(d)), d, label=file)
plt.legend(loc='lower left')
plt.show()
