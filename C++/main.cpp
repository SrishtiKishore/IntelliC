#include "csv_handler.cpp"
#include "data_analyzer.cpp"
#include "data_transform.cpp"
#include "neural_network.cpp"
#include "vector.cpp"
#include "matrix.cpp"
using namespace std;
int main(){
	try{
	CSVHandler c("train_handwriting.csv");
	vector <vector <double> > data;
	vector <string> label;
	c.readCSV(data, label);
	vector <vector <double> > X_train;
	vector <vector <double> > Y_train;
	vector <vector <double> > X_val;
	vector <vector <double> > Y_val;
	vector <int> rand = Vector::random_permutation(data.size());
	for(int i=0;i<450;i++){
		X_train.push_back(data[rand[i]-1]);
		vector <double> row (10, 0);
		row[(int)data[rand[i]-1][0]] = 1;
		Y_train.push_back(row);
	}
	for(int i=450;i<500;i++){
		X_val.push_back(data[rand[i]-1]);
		vector <double> row (10, 0);
		row[(int)data[rand[i]-1][0]] = 1;
		Y_val.push_back(row);
	}
	DataTransform <double> dt;
	X_train = dt.sliceColumn(X_train,1,785);
	X_val = dt.sliceColumn(X_val,1,785);
	/*DataAnalyzer <double> d(X_train);
	d.printData();
	DataAnalyzer <double> d2(Y_train);
	d2.printData();
	DataAnalyzer <double> d3(X_val);
	d3.preview();
	DataAnalyzer <double> d4(Y_val);
	d4.preview();*/
	vector <int> v;
	v.push_back(25);
	NeuralNetwork model(X_train, Y_train, v, 100, false);

	model.trainByGradientDescent(0.01, true, true);

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
		double mx2 = 0;
		double mx_idx2 = -1;
		for(int j=0;j<Y_val[i].size();j++){
			if(Y_val[i][j]>mx2){
				mx2 = Y_val[i][j];
				mx_idx2 = j;
			}
		}
		printf("%lf %lf\n",mx_idx, mx_idx2);
		if(fabs(Y_val[i][(int)mx_idx]-1)<0.000001){
			cnt += 1;
		}
	}
	printf("Accuracy - %lf\n",cnt/Y_val.size());
	}
catch(const char *s){
	printf("%s\n",s);
}
	return 0;
}