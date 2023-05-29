#include<iostream>
#include<bits/stdc++.h>
// #include<omp.h>
using namespace std;

const int n = 1e5+2;
bool vis[n];
vector<int> adj[n];

int main() {
    for(int i = 0; i < n; i++) {
        vis[i] = false;
    }

    int n, m;
    cout<<"Enter Vertex";
    cin >> n;
    cout<<"Enter edge";
    cin >> m;
    cout<<"Enter relation with space "<<endl;
    for(int i = 0; i < m; i++) {
        int x, y;
        cin >> x >> y;
        adj[x].push_back(y);
        adj[y].push_back(x);
    }

    queue<int> q;
    q.push(1);
    vis[1] = true;

    #pragma omp parallel
    {
        while(!q.empty()) {
            int node;
            #pragma omp critical
            {
                node = q.front();
                q.pop();
            }

            #pragma omp single
            cout << node << endl;

            #pragma omp for
            for(int i = 0; i < adj[node].size(); i++) {
                int neighbor = adj[node][i];
                #pragma omp critical
                {
                    if(!vis[neighbor]) {
                        vis[neighbor] = true;
                        #pragma omp critical
                        {
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    return 0;
}