nvcc four_checker_cuda.cu -O3 -D N=$1 -D M=$2 -o cuda_out
time -p ./cuda_out