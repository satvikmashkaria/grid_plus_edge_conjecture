#include <iostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;

int dist(int x1, int y1, int x2, int y2) { return abs(x1 - x2) + abs(y1 - y2); }

int distf(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4) {
    return min(dist(x1, y1, x2, y2),
               min(dist(x1, y1, x3, y3) + dist(x4, y4, x2, y2) + 1,
                   dist(x1, y1, x4, y4) + dist(x3, y3, x2, y2) + 1));
}

bool corner_check(int x, int y, int n, int m) {
    if (x == 1) {
        if (y == 1 || y == m) return false;
    } else if (x == n) {
        if (y == m || y == 1) return false;
    }
    return true;
}

bool four_cond(int x1, int y1, int x2, int y2, int n, int m) {
    int Gain1 = abs(abs(y1 - y2) - abs(x1 - x2)) - 1;
    bool first = corner_check(x1, y1, n, m) && corner_check(x2, y2, n, m);
    bool second = (Gain1 % 2 == 0) && (Gain1 > 0);
    bool third = 2 * min(abs(x2 - x1), abs(y1 - y2)) >= (Gain1 + 4);
    return first && second && third;
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

    for(int i = 0; i< num_pairs; i++){
        int x1 = ppx1[i], y1 = ppy1[i], x2 = ppx2[i], y2 = ppy2[i];
        conds[i] = four_cond(x1, y1, x2, y2, n, m);
    }

    for (int i = 0; i < ppx1.size(); i++) {
        int x1 = ppx1[i], y1 = ppy1[i], x2 = ppx2[i], y2 = ppy2[i];
        bool found = true;
        for (int j = 0; j < tpx1.size(); j++) {
            set<tuple<int, int, int>> distances;
            bool flag = true;
            for (int t = 0; t < n * m; t++) {
                int x_tmp = points_x[t], y_tmp = points_y[t];
                auto tup = make_tuple(
                    distf(x_tmp, y_tmp, tpx1[j], tpy1[j], x1, y1, x2, y2),
                    distf(x_tmp, y_tmp, tpx2[j], tpy2[j], x1, y1, x2, y2),
                    distf(x_tmp, y_tmp, tpx3[j], tpy3[j], x1, y1, x2, y2));
                if (distances.find(tup) != distances.end()) {
                    flag = false;
                    break;
                } else
                    distances.insert(tup);
            }
            if (flag) {
                found = false;
                break;
            }
        }
        if (!(conds[i] ^ found)) {
            if (found)
                cout << "MD is 4 when edge is between (" << x1 << "," << y1
                     << ") and (" << x2 << "," << y2 << ")\n";
        } else {
            cout << "Mistake\n";
            exit(-1);
        }
    }
    cout << "Success!\n";
}
