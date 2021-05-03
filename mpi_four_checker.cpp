#include<iostream>
#include<vector>
#include<utility>
#include<tuple>
#include<map>
#include<string>
#include<bits/stdc++.h>
#include<mpi.h>

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

int main(int argc, char* argv[]){
    if(argc < 3){
        cout<<"Please provide dimensions of the grid.\n";
        exit(-1);
    }




    MPI_Status status;
    int ierr,p,id;
    int down=1000;
    int up=999;
    int tag1=0;
    ierr = MPI_Init ( &argc, &argv );
    ierr = MPI_Comm_rank ( MPI_COMM_WORLD, &id );
    ierr = MPI_Comm_size ( MPI_COMM_WORLD, &p );




    int n = stoi(argv[1]);
    int m = stoi(argv[2]);
    vector< pair<int, int> > points;
    vector< pair< pair<int, int>, pair<int, int> > > point_pairs;
    vector< tuple<pair<int, int>, pair<int, int>, pair<int, int> > > three_points;
    cout<<id<<endl;

        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= m; j++){
                pair<int, int> p(i, j);
                points.push_back(p);
            }
        }


        for (int i = 0; i < points.size(); i++){
            for (int j = i+1; j < points.size(); j++){
                for(int k = j+1; k < points.size(); k++){
                    tuple< pair<int, int>, pair<int, int>, pair<int, int> > tup;
                    tup = make_tuple(points[i], points[j], points[k]);
                    three_points.push_back(tup);
                }
            }
        }
    

    if(id==0)
    {



        for (int i = 0; i < points.size(); i++){
            for (int j = i+1; j < points.size(); j++){
                pair<int, int> x = points[i], y = points[j];
                if(dist(x, y) > 1 && (x.first <= y.first) && (x.second >= y.second) && (x.second - y.second) >= (y.first - x.first))
                {
                    pair< pair<int, int>, pair<int, int> > p(points[i], points[j]);
                    point_pairs.push_back(p);
                }
            }
        }


        int x=three_points.size();
        int chunk_size=point_pairs.size()/p;
        for(int i=1;i<p;i++)
        {
            // MPI_Send(&x,1,MPI_INT,i,down,MPI_COMM_WORLD);
            // MPI_Send(&three_points[0],x*(sizeof(tuple<pair<int, int>, pair<int, int>, pair<int, int> >)),MPI_BYTE,i,down,MPI_COMM_WORLD);
            MPI_Send(&chunk_size,1,MPI_INT,i,down,MPI_COMM_WORLD);
            MPI_Send(&point_pairs[(i-1)*chunk_size],chunk_size*(sizeof(pair< pair<int, int>, pair<int, int> >)),MPI_BYTE,i,down,MPI_COMM_WORLD);

        }

        cout<<"chunk_size "<<chunk_size<<endl;
        for (int i = (p-1)*chunk_size; i < point_pairs.size(); i++){
            pair<int, int> x = point_pairs[i].first, y = point_pairs[i].second;
            
                bool found = true;
                for(int j = 0; j < three_points.size(); j++){
                    set< tuple <int, int, int> > distances;
                    bool flag = true;
                    for (int t = 0; t < points.size(); t++){
                        tuple<int, int, int> tup;
                        tup = make_tuple(distf(points[t], get<0>(three_points[j]), x, y), distf(points[t], get<1>(three_points[j]), x, y), distf(points[t], get<2>(three_points[j]), x, y));
                        if(distances.find(tup)!= distances.end()){
                            flag = false;
                            break;
                        }
                        else
                            distances.insert(tup);
                    }
                    if(flag){
                        found = false;
                        break;
                    }   
                }
                bool cond = four_cond(x, y, n, m);
                if(!(cond ^ found)){
                    if(found)
                        cout<<"MD is 4 when edge is between ("<<x.first<<","<<x.second<<") and ("<<y.first<<","<<y.second<<")"<<endl;
                }
                else{
                    cout<<"Mistake"<<endl;
                    exit(-1);
                }
            
        }
        cout<<"Success!\n";
    }
    else
    {
        // int x;
        // MPI_Recv(&x,1,MPI_INT,0,down,MPI_COMM_WORLD,&status);
        // three_points.resize(x);
        // MPI_Recv(&three_points[0],x*(sizeof(tuple<pair<int, int>, pair<int, int>, pair<int, int> >)),MPI_BYTE,0,down,MPI_COMM_WORLD,&status);
        
        int chunk_size;
        MPI_Recv(&chunk_size,1,MPI_INT,0,down,MPI_COMM_WORLD,&status);
        cout<<"chunk_size "<<chunk_size<<endl;
        point_pairs.resize(chunk_size);
        MPI_Recv(&point_pairs[0],chunk_size*(sizeof(pair< pair<int, int>, pair<int, int> >)),MPI_BYTE,0,down,MPI_COMM_WORLD,&status);


        cout<<"helo"<<endl;
        for (int i = 0; i < chunk_size; i++){
            pair<int, int> x = point_pairs[i].first, y = point_pairs[i].second;
            
            bool found = true;
            for(int j = 0; j < three_points.size(); j++){
                set< tuple <int, int, int> > distances;
                bool flag = true;
                for (int t = 0; t < points.size(); t++){
                    tuple<int, int, int> tup;
                    tup = make_tuple(distf(points[t], get<0>(three_points[j]), x, y), distf(points[t], get<1>(three_points[j]), x, y), distf(points[t], get<2>(three_points[j]), x, y));
                    if(distances.find(tup)!= distances.end()){
                        flag = false;
                        break;
                    }
                    else
                        distances.insert(tup);
                }
                if(flag){
                    found = false;
                    break;
                }   
            }
            bool cond = four_cond(x, y, n, m);
            if(!(cond ^ found)){
                if(found)
                    cout<<"MD is 4 when edge is between ("<<x.first<<","<<x.second<<") and ("<<y.first<<","<<y.second<<") "<<id<<endl;
            }
            else{
                cout<<"Mistake "<<id<<" "<<x.first<<" "<<x.second<<endl;

                // exit(-1);
            } 
        }
    }
    MPI_Finalize();

}