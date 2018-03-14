#include <bits/stdc++.h>
using namespace std;
#include "Cluster.cpp"
#include "Matrix.cpp"
#include "Vector.cpp"
int main(){
	vector <vector <double> > X = Matrix::prod(Matrix::random(100,4, true), 1000);
	Cluster model(X);
	vector <vector<double> > centroids;
	vector <int> v;
	double cost;
	model.cluster(10, centroids, v, true);

	for(int i=0;i<10;i++){
		for(int j=0;j<4;j++){
			printf("%lf ",centroids[i][j]);
		}
		printf("\n");
	}
	return 0;
}