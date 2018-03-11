#include "csv_handler.cpp"
#include "data_analyzer.cpp"
#include "data_transform.cpp"
#include "neural_network.cpp"
#include "vector.cpp"
#include "matrix.cpp"
using namespace std;
int main(){
	CSVHandler c("train_handwriting.csv");
	vector <vector <double> > data;
	vector <string> label;
	c.readCSV(data, label);
	vector <vector <double> > X_train;
	vector <vector <double> > Y_train;
	vector <vector <double> > X_val;
	vector <vector <double> > Y_val;
	for(int i=0;i<4500;i++){
		X_train.push_back(data[i]);
		vector <double> row (10, 0);
		row[(int)data[i][0]] = 1;
		Y_train.push_back(row);
	}
	for(int i=4500;i<data.size();i++){
		X_val.push_back(data[i]);
		vector <double> row (10, 0);
		row[(int)data[i][0]] = 1;
		Y_val.push_back(row);
	}
	printf("fvssv");

	DataTransform <double> dt;
	X_train = dt.sliceColumn(X_train,1,785);
	vector <int> v;
	v.push_back(25);
	NeuralNetwork model(X_train, Y_train, v, 100);
	model.trainByGradientDescent(0.01, true);
	vector <vector <double> > Y_p = model.predict(X_val);
	double cnt = 0; 
	for(int i=0;i<Y_p.size();i++){
		double mx = 0;
		double mx_idx = -1;
		for(int j=0;j<Y_p[i].size();j++){
			if(Y_p[i][j]>mx){
				mx = Y_p[i][j];
				mx_idx = j;
			}
		}
		if(fabs(Y_val[i][mx_idx]-1)<0.000001){
			cnt += 1;
		}
	}
	printf("Accuracy - %lf\n",cnt/Y_val.size());
	return 0;
}