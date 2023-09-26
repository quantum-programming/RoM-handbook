import os
import time
import numpy as np
from exputils.RoM.fwht import calculate_RoM_FWHT
from exputils.cover_states import make_cover_info

n_qubit = 14
rho_vec = np.random.rand(4**n_qubit)

make_cover_info(n_qubit)  # for the cache

start_time = time.perf_counter()
RoM = calculate_RoM_FWHT(n_qubit, rho_vec)[0]
end_time = time.perf_counter()

print("n_qubit: ", n_qubit)
print("RoM: ", RoM)
print("Time: ", end_time - start_time, "[s]")

with open(os.path.join(os.path.dirname(__file__), "time_of_fwht.csv"), mode="w") as f:
    f.write("n_qubit,RoM,time\n")
    f.write(f"{n_qubit},{RoM},{end_time - start_time}\n")
