import subprocess
from time import time
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns

m = 1
n_list = list(range(6, 17))


omp = []
subprocess.run(["g++", "-O3", "-fopenmp",
                "four_checker_omp.cpp", "-o", "omp_out"], stdout=subprocess.DEVNULL)
for n in n_list:
    for i in range(m):
        start = time()
        subprocess.run(["./omp_out", str(n), str(n)],
                       stdout=subprocess.DEVNULL)
        time_taken = time()-start
        print(f"OpenMP for {n}x{n}, time taken = {time_taken:.02f} sec")
        omp.append(time_taken)

mpi = []
subprocess.run(["mpic++", "-O3", "four_checker_mpi.cpp",
                "-o", "mpi_out"], stdout=subprocess.DEVNULL)
for n in n_list:
    for i in range(m):
        start = time()
        subprocess.run(["mpirun", "-np", "8", "--use-hwthread-cpus", "--allow-run-as-root",
                        "./mpi_out", str(n), str(n)], stdout=subprocess.DEVNULL)
        time_taken = time()-start
        print(f"MPI for {n}x{n}, time taken = {time_taken:.02f} sec")
        mpi.append(time_taken)

cuda = []
for n in n_list:
    subprocess.run(["nvcc", "four_checker_cuda.cu", "-O3",
                    f"-D N={n}",  "-o", "cuda_out"], stdout=subprocess.DEVNULL)
    for i in range(m):
        start = time()
        subprocess.run(["./cuda_out"], stdout=subprocess.DEVNULL)
        time_taken = time()-start
        print(f"CUDA for {n}x{n}, time taken = {time_taken:.02f} sec")
        cuda.append(time_taken)

cuda = np.array(cuda).reshape(-1, m).mean(1)
omp = np.array(omp).reshape(-1, m).mean(1)
mpi = np.array(mpi).reshape(-1, m).mean(1)

sns.set()
plt.figure(figsize=(12, 8))
plt.plot(n_list, omp, color='r', label="OpenMP")
plt.plot(n_list, mpi, color='b', label="MPI")
plt.plot(n_list, cuda, color='g', label="CUDA")
plt.legend()
plt.xlabel("Size of grid $n$")
plt.ylabel("Time take in sec")
plt.title("Timing Study")
plt.savefig("imgs/1.jpg")
