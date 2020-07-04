#include<iostream>
#include<vector>
#include<utility>
#include<tuple>
#include<map>
#include<string>
#include<pthread.h>

using namespace std;

int dist(pair<int, int> x, pair<int, int> y){
    return abs(x.first - y.first)+abs(x.second - y.second);
}

int distf(pair<int, int> x, pair<int, int> y, pair<int, int> x1, pair<int, int> y1){
    return min(dist(x, y), min(dist(x, x1)+dist(y1, y)+1, dist(x, y1)+dist(x1, y)+1));
}

bool corner_check(pair<int, int> x, int n, int m){
    pair<int, int> p(1, 1), q(1, m), r(n, m), s(n, 1);
    return x != p && x != q && x != r && x != s;
}

bool four_cond(pair<int, int> x1, pair<int, int> y1, int n, int m){
    int Gain1 = abs(abs(x1.second - y1.second) - abs(x1.first - y1.first)) - 1;
    
    bool first = corner_check(x1, n, m) && corner_check(y1, n, m);
    bool second = (Gain1 % 2 == 0) && (Gain1 > 0);
    bool third = 2 * min(abs(y1.first - x1.first), abs(x1.second - y1.second)) >= (Gain1 + 4);
    return first && second && third;
}

bool two_cond(pair<int, int> x1, pair<int, int> y1, int n, int m){
    int Gain1 = abs(abs(y1.second - x1.second) - abs(x1.first - y1.first)) - 1;
	int Gain = abs(y1.second - x1.second) + abs(x1.first - y1.first) - 1;

	bool first = Gain == 1;
	bool second = (!corner_check(x1, n, m) || !corner_check(y1, n, m)) && (Gain1 <= 1) && (Gain%2 == 1);
	bool third = (!corner_check(x1, n, m) || !corner_check(y1, n, m)) && (Gain1 >= 3) && (Gain%2 == 1) && (Gain - Gain1 <= 2);
	bool fourth = (!corner_check(x1, n, m) && !corner_check(y1, n, m)) && Gain%2 == 1;
	return first || second || third || fourth;
}

// int n, m;
vector< pair<int, int> > points;
vector< pair< pair<int, int>, pair<int, int> > > point_pairs;

struct args {
    pair<int, int> x;
    pair<int, int> y; 
    int n; 
    int m; 
    // vector< pair<int, int> > points;
    // vector< pair< pair<int, int>, pair<int, int> > > point_pairs;
};

void *two_checker(void *a){
    struct args *my_args;
    my_args = (struct args *)a;

    pair<int, int> x = my_args->x;
    pair<int, int> y = my_args->y; 
    int n = my_args->n; 
    int m = my_args->m; 
    // vector< pair<int, int> > points = my_args->points;
    // vector< pair< pair<int, int>, pair<int, int> > > point_pairs = my_args->point_pairs;
    bool found = false;
    for(int j = 0; j < point_pairs.size(); j++){
        map< tuple <int, int>, int> distances;
        bool flag = true;
        for (int t = 0; t < points.size(); t++){
            tuple<int, int> tup;
            tup = make_tuple(distf(points[t], get<0>(point_pairs[j]), x, y), distf(points[t], get<1>(point_pairs[j]), x, y));
            if(distances.count(tup) > 0){
                flag = false;
                break;
            }
            else
                distances[tup] = 1;
        }
        if(flag){
            found = true;
            break;
        }   
    }
    bool cond = two_cond(x, y, n, m);
    if(!(cond ^ found)){
        if(found)
            cout<<1;
            // cout<<"MD is 2 when edge is between "<<x.first<<" "<<x.second<<" and "<<y.first<<" "<<y.second<<endl;
    }
    else{
        cout<<"Mistake\n";
        exit(1);
    }
    // pthread_exit(NULL);
    return 0;
}


int main(){
    int m = 10, n = 10;

    // vector< pair<int, int> > points;

    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= m; j++){
            pair<int, int> p(i, j);
            points.push_back(p);
        }
    }

    // vector< pair< pair<int, int>, pair<int, int> > > point_pairs;

    for (int i = 0; i < points.size(); i++){
        for (int j = i+1; j < points.size(); j++){
            pair< pair<int, int>, pair<int, int> > p(points[i], points[j]);
            point_pairs.push_back(p);
        }
    }

    // vector< tuple<pair<int, int>, pair<int, int>, pair<int, int> > > three_points;

    // for (int i = 0; i < points.size(); i++){
    //     for (int j = i+1; j < points.size(); j++){
    //         for(int k = j+1; k < points.size(); k++){
    //             tuple< pair<int, int>, pair<int, int>, pair<int, int> > tup;
    //             tup = make_tuple(points[i], points[j], points[k]);
    //             three_points.push_back(tup);
    //         }
    //     }
    // }
            
    cout<<"done\n";
    vector<pthread_t> threads;
    for (int i = 0; i < point_pairs.size(); i++){
        pair<int, int> x = point_pairs[i].first, y = point_pairs[i].second;
        if(dist(x, y) > 1 && (x.first <= y.first) && (x.second >= y.second) && (x.second - y.second) >= (y.first - x.first)){
            pthread_t thread;
            struct args a;
            a.m = m;
            a.n = n;
            a.x = x;
            a.y = y;
            // a.points = points;
            // a.point_pairs = point_pairs;
            two_checker((void *)&a);
            // threads.push_back(thread);
            // int rc = pthread_create(&thread, NULL, two_checker, (void *)&a);
            // if (rc) {
                // cout << "Error:unable to create thread," << rc << endl;
                // exit(-1);
            // }
        }
    }
    void* status;
    for(int i = 0; i < threads.size(); i++){
        pthread_join(threads[i], &status);
    }
    // pthread_exit(NULL);
    cout<<"Successful\n";
}