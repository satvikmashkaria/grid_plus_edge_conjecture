g++ -O3 -fopenmp four_checker_omp.cpp -o omp_out
time -p ./omp_out $1 $2