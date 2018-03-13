
#ifndef _VECTOR_
	#include <vector>
	#define _VECTOR_
#endif

#ifndef _TIME_H_
	#include <time.h>
	#define _TIME_H_
#endif

#ifndef _STDIO_H_
	#include <stdio.h>
	#define _STDIO_H_
#endif

#ifndef _UTILITY_
	#include <utility>
	#define _UTILITY_
#endif


#include "Vector.cpp"
#include "Matrix.cpp"
#include "Datatransform.cpp"


class DeepNeuralNetwork{
private:
	vector <vector <double> > X;
	vector <vector <double> > X_t;
	vector <vector <double> > Y;
	vector <vector <vector <double> > > theta;
	vector <vector <vector <double> > > neurons; //Triple dimension: This is because there are 'm' neural network each having same theta.

	/*
		0th layer is the Matrix A itself. 
		Total number of layers = hidden layer count + 2
		theta[i][j][k] = Between the ith and (i+1)th layer. j is in (i+1)th layer. k is in ith layer.
		neurons[i][j][k] = i is the ith layer. 0th layer is A itself. j is the training data number and k is the feature number
	*/
	vector <double> avg;
	vector <double> std;
	int n;
	int m;
	int input_layer_size;
	int output_layer_size;
	int hidden_layer_count;
	int layer_count;
	int input_layer;
	int output_layer;
	double lambda;
	bool isNormalized;
	DataTransform <double> _d;
	double sigmoid(double z){
		return 1.0/(1+exp(-z));
	}
	static double randDouble(bool positiveOnly = false){
		if(rand()%2==0 || positiveOnly)
			return (double)rand()/(double)RAND_MAX;
		else
			return -(double)rand()/(double)RAND_MAX;
	}
	vector <vector<double> > hypothesis(){
		return neurons[output_layer];
	}
	double cost(){
		vector <vector<double> > h = hypothesis();
		vector <double> v1 = Matrix::sum(Matrix::prod(Y, Matrix::log(h)));
		vector <double> v2 = Matrix::sum(Matrix::prod(Matrix::diff(Matrix::ones(Y.size(), Y[0].size()) , Y),Matrix::log(Matrix::diff(Matrix::ones(h.size(), h[0].size()) , h))));
		double reg = 0;
		for(int i=0;i<theta.size();i++){
			for(int j=0;j<theta[i].size();j++){
				for(int k=1;k<theta[i][j].size();k++){
					reg += theta[i][j][k]*theta[i][j][k];
				}
			}
		}
		return (-1.0/m)*Vector::sum(Vector::sum(v1,v2)) + (lambda/(2*m))*reg;
	}
	void forwardPropagate(){
		for(int i=1;i<=output_layer;i++){
			neurons[i] = Matrix::sigmoid(Matrix::multiply(neurons[i-1], Matrix::transpose(theta[i-1])));
			if(i!=output_layer)
				_d.prependColumn(neurons[i],Vector::ones(m));
		}
	}
	void backPropagate(vector <vector <vector <double> > > &Delta){
		vector <vector <double> > h = hypothesis();
		for(int i=0;i<m;i++){
			vector <vector <double> > del;
			del.push_back(Vector::prod(Vector::diff(h[i],Y[i]),Vector::prod(h[i],Vector::diff(Vector::ones(h[i].size()),h[i]))));
			int last_del = 0;
			for(int j=output_layer-1;j>=1;j--){
				vector <double> v = Vector::prod(Matrix::multiply(Matrix::transpose(theta[j]),del[last_del]),Vector::prod(neurons[j][i],Vector::diff(Vector::ones(neurons[j][i].size()),neurons[j][i])));
				reverse(v.begin(),v.end());
				v.pop_back();
				reverse(v.begin(),v.end());
				del.push_back(v);
				last_del++;
			}
			reverse(del.begin(),del.end());
			for(int j=0;j<output_layer;j++){
				Delta[j] = Matrix::sum(Delta[j], Matrix::multiply(Vector::upgrade(del[j]),Matrix::transpose(Vector::upgrade(neurons[j][i]))));
			}
		}

	}
public:
	DeepNeuralNetwork(const vector <vector <double> > &data, const vector <vector<double> > &label, const vector <int> &hidden_units, double l, bool normalize = false){
		if(data.size()==0) throw "Data must not be empty";
		if(data.size()!=label.size())	throw "Number of X and y must match\n";
		if(!Matrix::isMatrix(data))	throw "X must be a matrix (i.e double dimensional vector)\n";
		if(!Matrix::isMatrix(label))	throw "Y must be a matrix (i.e double dimensional vector)\n";

		m = data.size();
		n = data[0].size();
		input_layer_size = n+1;
		output_layer_size = label[0].size();
		hidden_layer_count = hidden_units.size();
		layer_count = hidden_layer_count + 2;
		input_layer = 0;
		output_layer = layer_count-1;
		lambda = l;

		for(int i=0;i<m;i++){
			int cnt = 0;
			for(int j=0;j<output_layer_size;j++){
				if(fabs(label[i][j]-1)>0.000001 && fabs(label[i][j])>0.000001){
					throw "Y must be either 0 or 1.\n";
				}

				if(fabs(label[i][j]-1)<=0.000001){
					cnt++;
				}
			}
			if(cnt!=1){
				throw "Each X must have exactly one label.\n";
			}
		}
		if(normalize){
			avg = Matrix::avg(data);
			std = Matrix::std(data);
			X = Matrix::normalize(data);
			isNormalized = true;
		}
		else{
			X = data;
			isNormalized = false;
		}

		_d.prependColumn(X,Vector::ones(m));
		X_t = Matrix::transpose(X);

		Y = label;

		int curr = input_layer_size;
		for(int l=0;l<hidden_layer_count;l++){
			theta.push_back(Matrix::random(hidden_units[l], curr));
			curr = 1+hidden_units[l];
		}
		theta.push_back(Matrix::random(output_layer_size, curr));


		vector <vector<double> > v;
		neurons.push_back(X);
		for(int i=1;i<=hidden_layer_count;i++){
			v = Matrix::sigmoid(Matrix::multiply(neurons[i-1], Matrix::transpose(theta[i-1])));
			_d.prependColumn(v,Vector::ones(m));
			neurons.push_back(v);
		}
		v = Matrix::sigmoid(Matrix::multiply(neurons[output_layer-1], Matrix::transpose(theta[output_layer-1])));
		neurons.push_back(v);
	}
	double trainByGradientDescent(double alpha,  bool gradientCheck, bool printCost = false){
		if(!isNormalized){
			printf("Gradient Descent without Normalization may take a long time to complete. Try to normalize the features for faster descent.\n");
		}
		srand(time(0));
		for(int i=0;i<theta.size();i++){
			for(int j=0;j<theta[i].size();j++){
				for(int k=0;k<theta[i][j].size();k++){
					theta[i][j][k] = randDouble()/10000;
				}
			}
		}
		double prev_cost = 1000000*m, curr_cost;
		while(true){

			forwardPropagate();

			vector <vector<vector<double> > > Delta  = theta;
			for(int i=0;i<Delta.size();i++){
				for(int j=0;j<Delta[i].size();j++){
					for(int k=0;k<Delta[i][j].size();k++){
						Delta[i][j][k] = 0;
					}
				}
			}

			backPropagate(Delta);

			for(int i=0;i<Delta.size();i++){
				for(int j=0;j<Delta[i].size();j++){
					for(int k=0;k<Delta[i][j].size();k++){
						if(gradientCheck){
							double orig = theta[i][j][k];
							theta[i][j][k] += 0.0001;
							forwardPropagate();
							double c1 = cost();
							theta[i][j][k] -= 0.0002;
							forwardPropagate();
							double c2 = cost();
							if(k==0)
								printf("Gradient Checking - %lf %lf\n",(c1-c2)/0.0002,(1.0/m)*Delta[i][j][k]);
							else
								printf("Gradient Checking - %lf %lf\n",(c1-c2)/0.0002,(1.0/m)*(Delta[i][j][k] + lambda*orig));
							//Restore
							theta[i][j][k] += 0.0001;
							forwardPropagate();
						}
						if(k==0)
							theta[i][j][k] = theta[i][j][k] - (alpha/m)*Delta[i][j][k];
						else
							theta[i][j][k] = theta[i][j][k] - (alpha/m)*(Delta[i][j][k] + lambda*theta[i][j][k]);
							
					}
				}
			}

			curr_cost = cost();
			if(curr_cost-prev_cost>0 || fabs(curr_cost-prev_cost)<0.000001){
				if(curr_cost-prev_cost > 0.000001){
					printf("A overshoot was observed during learning. Try to decrease the learning rate or increase the layers.\n");
				}
				break;
			}
			if(printCost)
				printf("Cost: %lf Difference: %lf\n",curr_cost,prev_cost-curr_cost);
			prev_cost = curr_cost;
		}
		return prev_cost;
	}

	vector <vector<double> > predict(vector <vector<double> > X_p){ //Pass a copy since we will modify it.
		// The matrix is automatically verified by rest of the functions.
		if(isNormalized){
			// Renormalize using the same mean and standard deviation.
			for(int i=0;i<n;i++){
				for(int j=0;j<X_p.size();j++){
					X_p[j][i] -= avg[i];
					X_p[j][i] /= std[i];
				}
			}
		}
		_d.prependColumn(X_p,Vector::ones(X_p.size()));
		for(int i=0;i<theta.size();i++){
			printf("Layer %d\n",i);
			for(int j=0;j<theta[i].size();j++){
				for(int k=0;k<theta[i][j].size();k++){
					printf("%d %d %lf\n",k,j,theta[i][j][k]);
				}
			}
		}
		//Forward propagate X_p
		for(int i=1;i<=hidden_layer_count;i++){
			X_p = Matrix::sigmoid(Matrix::multiply(X_p, Matrix::transpose(theta[i-1])));
			_d.prependColumn(X_p,Vector::ones(X_p.size()));
		}
		X_p = Matrix::sigmoid(Matrix::multiply(X_p, Matrix::transpose(theta[output_layer-1])));
		return X_p;
	}
};