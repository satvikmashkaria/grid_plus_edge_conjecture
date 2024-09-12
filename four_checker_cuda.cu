#include <iostream>
#include <vector>

#define TPB 256

#ifndef N
#define N 10
#endif
#ifndef M
#define M N
#endif

using namespace std;

__host__ __device__ int dist(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

__device__ int distf(int x1, int y1, int x2, int y2, int x3, int y3, int x4,
                     int y4) {
    return min(dist(x1, y1, x2, y2),
               min(dist(x1, y1, x3, y3) + dist(x4, y4, x2, y2) + 1,
                   dist(x1, y1, x4, y4) + dist(x3, y3, x2, y2) + 1));
}

__device__ bool corner_check(int x, int y) {
    if ((x == 1 || x == N) && (y == 1 || y == M))
        return false;
    else
        return true;
}

__global__ void four_cond(int* x1, int* y1, int* x2, int* y2, int* num_pairs,
                          bool* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *num_pairs) return;
    int Gain1 = abs(abs(y1[idx] - y2[idx]) - abs(x1[idx] - x2[idx])) - 1;
    bool first =
        corner_check(x1[idx], y1[idx]) && corner_check(x2[idx], y2[idx]);
    bool second = (Gain1 % 2 == 0) && (Gain1 > 0);
    bool third =
        2 * min(abs(x2[idx] - x1[idx]), abs(y1[idx] - y2[idx])) >= (Gain1 + 4);
    out[idx] = first && second && third;
    return;
}

__global__ void make_three_points(int* px, int* py, int* tpx1, int* tpy1,
                                  int* tpx2, int* tpy2, int* tpx3, int* tpy3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int ntp = (N * M * (N * M - 1) * (N * M - 2)) / 6;
    if (i >= ntp) return;
    int c = 0;
    for (int i = 0; i < N * M; i++) {
        for (int j = i + 1; j < N * M; j++) {
            for (int k = j + 1; k < N * M; k++) {
                tpx1[c] = px[i];
                tpy1[c] = py[i];
                tpx2[c] = px[j];
                tpy2[c] = py[j];
                tpx3[c] = px[k];
                tpy3[c] = py[k];
                c++;
            }
        }
    }
    return;
}

__device__ constexpr int get_upper(int b) {
    int a = 1;
    while (a <= b) a <<= 1;
    return a;
}

__global__ void find(int* ppx1, int* ppy1, int* ppx2, int* ppy2, int* tpx1,
                     int* tpy1, int* tpx2, int* tpy2, int* tpx3, int* tpy3,
                     int* px, int* py, int* npp_dev, bool* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *npp_dev) return;
    const int ntp = (N * M * (N * M - 1) * (N * M - 2)) / 6;
    const int x1 = ppx1[i], y1 = ppy1[i], x2 = ppx2[i], y2 = ppy2[i];
    int x_tmp, y_tmp, hash, hash_small1, hash_small2;
    bool found = true;

    const int m = N + M - 1;
    const int m2 = m * m;
    const int hash_size = get_upper(5 * m);

    bool hash_table1[hash_size];
    bool hash_table2[hash_size];
    int distances[N * M];

    int a, b, c;

    for (int j = 0; j < ntp; j++) {
        for (int k = 0; k < hash_size; k++) {
            hash_table1[k] = false;
            hash_table2[k] = false;
        }
        bool flag = true;
        for (int t = 0; t < N * M; t++) {
            x_tmp = px[t];
            y_tmp = py[t];
            a = distf(x_tmp, y_tmp, tpx1[j], tpy1[j], x1, y1, x2, y2);
            b = distf(x_tmp, y_tmp, tpx2[j], tpy2[j], x1, y1, x2, y2);
            c = distf(x_tmp, y_tmp, tpx3[j], tpy3[j], x1, y1, x2, y2);
            hash = a * m2 + b * m + c;
            hash_small1 = (5 * a) ^ (3 * b) ^ (1 * c);
            hash_small2 = (3 * a) ^ (1 * b) ^ (5 * c);
            if (hash_table1[hash_small1] && hash_table2[hash_small2]) {
                for (int k = 0; k < t; k++) {
                    if (distances[k] == hash) {
                        flag = false;
                        break;
                    }
                }
                if (!flag) break;
            }
            hash_table1[hash_small1] = true;
            hash_table2[hash_small2] = true;
            distances[t] = hash;
        }
        if (flag) {
            found = false;
            break;
        }
    }
    out[i] = found;
    return;
}

int main(int argc, char* argv[]) {
    int points_x[N * M], points_y[N * M];
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= M; j++) {
            points_x[M * (i - 1) + j - 1] = i;
            points_y[M * (i - 1) + j - 1] = j;
        }
    }

    vector<int> ppx1, ppy1, ppx2, ppy2;
    for (int i = 0; i < N * M; i++) {
        for (int j = i + 1; j < N * M; j++) {
            int x1 = points_x[i], y1 = points_y[i], x2 = points_x[j],
                y2 = points_y[j];
            if (dist(x1, y1, x2, y2) > 1 && (x1 <= x2) && (y1 >= y2) &&
                (y1 - y2) >= (x2 - x1)) {
                ppx1.push_back(x1);
                ppy1.push_back(y1);
                ppx2.push_back(x2);
                ppy2.push_back(y2);
            }
        }
    }

    int num_pairs = ppx1.size();
    bool conds[num_pairs];
    bool* cond_dev;
    int cond_size = num_pairs * sizeof(bool);
    int pp_size = num_pairs * sizeof(int);
    int num_blocks = num_pairs / TPB;
    int *ppx1_dev, *ppx2_dev, *ppy1_dev, *ppy2_dev, *npp_dev;
    num_blocks += (num_pairs % TPB > 0);

    cudaMalloc((void**)&npp_dev, sizeof(int));
    cudaMalloc((void**)&cond_dev, cond_size);
    cudaMalloc((void**)&ppx1_dev, pp_size);
    cudaMalloc((void**)&ppy1_dev, pp_size);
    cudaMalloc((void**)&ppx2_dev, pp_size);
    cudaMalloc((void**)&ppy2_dev, pp_size);

    cudaMemcpy(npp_dev, &num_pairs, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ppx1_dev, &ppx1[0], pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ppy1_dev, &ppy1[0], pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ppx2_dev, &ppx2[0], pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ppy2_dev, &ppy2[0], pp_size, cudaMemcpyHostToDevice);

    four_cond<<<num_blocks, TPB>>>(ppx1_dev, ppy1_dev, ppx2_dev, ppy2_dev,
                                   npp_dev, cond_dev);

    cudaMemcpy(conds, cond_dev, cond_size, cudaMemcpyDeviceToHost);

    int *tpx1_dev, *tpx2_dev, *tpx3_dev, *tpy1_dev, *tpy2_dev, *tpy3_dev;
    int num_tp = (N * M * (N * M - 1) * (N * M - 2)) / 6;
    int tp_size = num_tp * sizeof(int);

    bool founds[num_pairs];
    bool* founds_dev;
    int p_size = N * M * sizeof(int);
    int *px_dev, *py_dev;

    cudaMalloc((void**)&founds_dev, cond_size);
    cudaMalloc((void**)&px_dev, p_size);
    cudaMalloc((void**)&py_dev, p_size);
    cudaMemcpy(px_dev, points_x, p_size, cudaMemcpyHostToDevice);
    cudaMemcpy(py_dev, points_y, p_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&tpx1_dev, tp_size);
    cudaMalloc((void**)&tpx2_dev, tp_size);
    cudaMalloc((void**)&tpx3_dev, tp_size);
    cudaMalloc((void**)&tpy1_dev, tp_size);
    cudaMalloc((void**)&tpy2_dev, tp_size);
    cudaMalloc((void**)&tpy3_dev, tp_size);

    make_three_points<<<1, 1>>>(px_dev, py_dev, tpx1_dev, tpy1_dev, tpx2_dev,
                                tpy2_dev, tpx3_dev, tpy3_dev);

    find<<<num_blocks, TPB>>>(ppx1_dev, ppy1_dev, ppx2_dev, ppy2_dev, tpx1_dev,
                              tpy1_dev, tpx2_dev, tpy2_dev, tpx3_dev, tpy3_dev,
                              px_dev, py_dev, npp_dev, founds_dev);

    cudaMemcpy(founds, founds_dev, cond_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_pairs; i++) {
        if (!(conds[i] ^ founds[i])) {
            if (founds[i]) {
                int x1 = ppx1[i], y1 = ppy1[i], x2 = ppx2[i], y2 = ppy2[i];
                cout << "MD is 4 when edge is between (" << x1 << "," << y1
                     << ") and (" << x2 << "," << y2 << ")\n";
            }
        } else {
            cout << "Mistake\n";
            exit(-1);
        }
    }
    cout << "Success!\n";
}
