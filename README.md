# Testing a conjecture for Metric Dimension of Grid + Edge with Parallel Programming

## ME 766: Final Project

This repository contains the code for final project of ME 766 course @ IIT Bombay. The team members are:
- Mohd Safwan
- Kushal Patil
- Satvik Mashkaria
- Sakhsam Khandelwal

![](conjecture.png)

For a mxn grid augmented with one edge, to check that there is no resolving set of size 3 if and only if the MD=4 condition in our conjecture hold, run any of the following.
```bash
./omp.sh m n
./mpi.sh m n
./cuda.sh m n
```

