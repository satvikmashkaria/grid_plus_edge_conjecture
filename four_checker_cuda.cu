#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define TPB 256

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

__device__ bool corner_check(int x, int y, int n, int m) {
    if (x == 1) {
        if (y == 1 || y == m) return false;
    } else if (x == n) {
        if (y == m || y == 1) return false;
    }
    return true;
}

__global__ void four_cond(int* x1, int* y1, int* x2, int* y2, int n, int m,
                          int num_pairs, bool* out) {
    int idx = blockIdx.x * TPB + threadIdx.x;
    if(idx >= num_pairs) return;
    int Gain1 = abs(abs(y1[idx] - y2[idx]) - abs(x1[idx] - x2[idx])) - 1;
    bool first = corner_check(x1[idx], y1[idx], n, m) &&
                 corner_check(x2[idx], y2[idx], n, m);
    bool second = (Gain1 % 2 == 0) && (Gain1 > 0);
    bool third =
        2 * min(abs(x2[idx] - x1[idx]), abs(y1[idx] - y2[idx])) >= (Gain1 + 4);
    out[idx] = first && second && third;
    return;
}

__global__ void find(int* ppx1, int* ppy1, int* ppx2, int* ppy2, int* tpx1,
                     int* tpy1, int* tpx2, int* tpy2, int* tpx3, int* tpy3,
                     int* px, int* py, int n, int m, int ntp, int npp, bool* out) {

    int i = blockIdx.x * TPB + threadIdx.x;
    if(i >= npp) return;
    const int NM = 20 * 20;
    int M = max(m, n) + 1;
    bool found = true;
    int x1 = ppx1[i], y1 = ppy1[i], x2 = ppx2[i], y2 = ppy2[i];
    for (int j = 0; j < ntp; j++) {
        int distances[NM] = {0};
        bool flag = true;
        for (int t = 0; t < n * m; t++) {
            int x_tmp = px[t], y_tmp = py[t];
            int hash =
                distf(x_tmp, y_tmp, tpx1[j], tpy1[j], x1, y1, x2, y2) * M * M +
                distf(x_tmp, y_tmp, tpx2[j], tpy2[j], x1, y1, x2, y2) * M +
                distf(x_tmp, y_tmp, tpx3[j], tpy3[j], x1, y1, x2, y2);
            for (int k = 0; k < t; k++) {
                if (hash == distances[k]) {
                    flag = false;
                    break;
                }
            }
            distances[t] = hash;
            if (!flag) break;
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
    if (argc < 3) {
        cout << "Please provide dimensions of the grid.\n";
        exit(-1);
    }
    int n = stoi(argv[1]);
    int m = stoi(argv[2]);

    int points_x[n * m], points_y[n * m];
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            points_x[m * (i - 1) + j - 1] = i;
            points_y[m * (i - 1) + j - 1] = j;
        }
    }

    vector<int> ppx1, ppy1, ppx2, ppy2;
    for (int i = 0; i < n * m; i++) {
        for (int j = i + 1; j < n * m; j++) {
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

    vector<int> tpx1, tpy1, tpx2, tpy2, tpx3, tpy3;
    for (int i = 0; i < n * m; i++) {
        for (int j = i + 1; j < n * m; j++) {
            for (int k = j + 1; k < n * m; k++) {
                tpx1.push_back(points_x[i]);
                tpy1.push_back(points_y[i]);
                tpx2.push_back(points_x[j]);
                tpy2.push_back(points_y[j]);
                tpx3.push_back(points_x[k]);
                tpy3.push_back(points_y[k]);
            }
        }
    }

    int num_pairs = ppx1.size();
    bool conds[num_pairs];
    bool* cond_dev;
    int *ppx1_dev, *ppx2_dev, *ppy1_dev, *ppy2_dev;
    int cond_size = num_pairs * sizeof(bool);
    int pp_size = num_pairs * sizeof(int);

    cudaMalloc((void**)&cond_dev, cond_size);
    cudaMalloc((void**)&ppx1_dev, pp_size);
    cudaMalloc((void**)&ppy1_dev, pp_size);
    cudaMalloc((void**)&ppx2_dev, pp_size);
    cudaMalloc((void**)&ppy2_dev, pp_size);

    cudaMemcpy(ppx1_dev, &ppx1[0], pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ppy1_dev, &ppy1[0], pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ppx2_dev, &ppx2[0], pp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ppy2_dev, &ppy2[0], pp_size, cudaMemcpyHostToDevice);

    four_cond<<<num_pairs / TPB, TPB>>>(ppx1_dev, ppy1_dev, ppx2_dev, ppy2_dev,
                                        n, m, num_pairs, cond_dev);

    cudaMemcpy(conds, cond_dev, cond_size, cudaMemcpyDeviceToHost);

    int *tpx1_dev, *tpx2_dev, *tpx3_dev, *tpy1_dev, *tpy2_dev, *tpy3_dev;
    int num_tp = tpx1.size();
    int tp_size = num_tp * sizeof(int);

    cudaMalloc((void**)&tpx1_dev, tp_size);
    cudaMalloc((void**)&tpx2_dev, tp_size);
    cudaMalloc((void**)&tpx3_dev, tp_size);
    cudaMalloc((void**)&tpy1_dev, tp_size);
    cudaMalloc((void**)&tpy2_dev, tp_size);
    cudaMalloc((void**)&tpy3_dev, tp_size);

    cudaMemcpy(tpx1_dev, &tpx1[0], tp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tpy1_dev, &tpy1[0], tp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tpx2_dev, &tpx2[0], tp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tpy2_dev, &tpy2[0], tp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tpx3_dev, &tpx3[0], tp_size, cudaMemcpyHostToDevice);
    cudaMemcpy(tpy3_dev, &tpy3[0], tp_size, cudaMemcpyHostToDevice);

    bool founds[num_pairs];
    bool* founds_dev;
    int p_size = n * m * sizeof(int);
    int *px_dev, *py_dev;
    cudaMalloc((void**)&founds_dev, cond_size);
    cudaMalloc((void**)&px_dev, p_size);
    cudaMalloc((void**)&py_dev, p_size);
    cudaMemcpy(px_dev, points_x, p_size, cudaMemcpyHostToDevice);
    cudaMemcpy(py_dev, points_y, p_size, cudaMemcpyHostToDevice);

    find<<<num_pairs / TPB, TPB>>>(
        ppx1_dev, ppy1_dev, ppx2_dev, ppy2_dev, tpx1_dev, tpy1_dev, tpx2_dev,
        tpy2_dev, tpx3_dev, tpy3_dev, px_dev, py_dev, n, m, num_tp, num_pairs, founds_dev);

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
