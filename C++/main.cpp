#include <bits/stdc++.h>
using namespace std;
#include "Cluster.cpp"
#include "Matrix.cpp"
#include "Vector.cpp"
int main(){
	vector <vector <double> > X = Matrix::random(100,4);
	Cluster model(X);
	vector <vector<double> > centroids;
	vector <int> v;
	double cost;
	model.cluster(10, centroids, v, cost, true);

	for(int i=0;i<100;i++){
		printf("%d\n",v[i]);
	}
	return 0;
}