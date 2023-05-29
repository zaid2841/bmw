#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

void dfs(int node, vector<vector<int>>& graph, vector<bool>& visited)
{
    stack<int> st;
    st.push(node);

    while (!st.empty()) {
        node = st.top();
        st.pop();

        if (!visited[node]) {
            visited[node] = true;
            cout << "Visited node: " << node << endl;

            // Push adjacent unvisited nodes onto the stack
            #pragma omp parallel for
            for (size_t i = 0; i < graph[node].size(); ++i) {
                int neighbor = graph[node][i];
                if (!visited[neighbor]) {
                    st.push(neighbor);
                }
            }
        }
    }
}

int main()
{
    vector<vector<int>> graph = {
        {1, 2},     // Node 0 is connected to nodes 1 and 2
        {0, 3, 4},  // Node 1 is connected to nodes 0, 3, and 4
        {0, 5},     // Node 2 is connected to nodes 0 and 5
        {1},        // Node 3 is connected to node 1
        {1},        // Node 4 is connected to node 1
        {2}         // Node 5 is connected to node 2
    };

    vector<bool> visited(graph.size(), false);

    #pragma omp parallel for
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!visited[i]) {
            dfs(i, graph, visited);
        }
    }

    return 0;
}