#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <vector>

#include "eigen-eigen\eigen-eigen-b9cd8366d4e8\Eigen\Dense"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

#define MAX_INPUTDATA_LEN 1000
#define MAX_DATA_COUNT 100000
#define DIMENSION 14

#define NUMBER_OF_NODE 14
#define MAX_NUMBER_OF_NODE 14
#define NUMBER_OF_HIDDEND_LAYER 3
#define MAX_NUMBER_OF_HIDDEND_LAYER 10
#define EPOCH 500
#define RBM_EPOCH 300

#define LEARNING_RATE 0.0005
#define RBM_LEARNING_RATE 0.0005
#define MOMENTUM 0.8
#define RANDOM_RANGE 0.0001

void setRandFill(MatrixXd *mat)
{
	srand(time(NULL));
	for (int i = 0; i < NUMBER_OF_NODE; i++)
	{
		for (int j = 0; j < NUMBER_OF_NODE; j++)
		{
			double randvalue = (double)(rand() % 100)*RANDOM_RANGE;
			if (rand() % 10 > 5)
				randvalue = -randvalue;
			(*mat)(i, j) = randvalue;
		}
	}
}

void setRandFill(MatrixXd &mat)
{
	srand(time(NULL));
	for (int i = 0; i < NUMBER_OF_NODE; i++)
	{
		for (int j = 0; j < DIMENSION; j++)
		{
			double randvalue = (double)(rand()*rand() % 100)*RANDOM_RANGE;
			if (rand() % 10 > 5)
				randvalue = -randvalue;
			mat(i, j) = randvalue;
		}
	}
}
void setRandFill(VectorXd &vec)
{
	srand(time(NULL));
	for (int i = 0; i < NUMBER_OF_NODE; i++)
	{
		double randvalue = (double)(rand()*rand()*rand() % 100)*RANDOM_RANGE;
		if (rand() % 10 > 5)
			randvalue = -randvalue;
		vec(i) = randvalue;
	}
}

void swap(int & a, int & b)
{
	int tmp;
	tmp = a;
	a = b;
	b = tmp;
}

vector<int> rand_order_gen(int len) {

	vector<int> a;
	for (int i = 0; i < len; i++)
		a.push_back(i);
	random_shuffle(a.begin(), a.end());

	return a;
}

VectorXd array2vector(double * arr) {
	VectorXd ret(DIMENSION);
	for (int i = 0; i < DIMENSION; i++)
	{
		ret(i) = arr[i];
	}
	return ret;
}

VectorXd matrix2vector(MatrixXd mat, int col_num)
{
	VectorXd ret(DIMENSION);
	ret = mat.col(col_num);
	return ret;
}

double Sigmoid(double input)
{
	return 1.0 / (1.0 + exp(-input));
}

void TEST(MatrixXd ** W, MatrixXd first_W, VectorXd V, double ** datum_t, int * lable_t, int N) {
	/*test*/
	double result;
	int correct = 0;
	char inputdata[MAX_DATA_COUNT];
	MatrixXd hidden(NUMBER_OF_HIDDEND_LAYER, NUMBER_OF_NODE);
	stringstream stream;
	ofstream log("log.txt", ios::app);
	double output;

	VectorXd h_vec(NUMBER_OF_NODE);


	for (int t = 0; t < N; t++)
	{
		/*DATA SET*/
		VectorXd input = array2vector(datum_t[t]);
		double r = lable_t[t];

		/*feed forward h*/
		hidden(0, 0) = 1;
		for (int i = 1; i < NUMBER_OF_NODE; i++) {
			VectorXd first_W_i(DIMENSION);
			first_W_i = first_W.row(i);
			double O = first_W_i.dot(input);
			hidden(0, i) = Sigmoid(O);
		}
		for (int k = 1; k < NUMBER_OF_HIDDEND_LAYER; k++)
		{
			VectorXd W_i(NUMBER_OF_NODE);
			hidden(k, 0) = 1;
			for (int i = 1; i < NUMBER_OF_NODE; i++) {
				W_i = (*W[k]).row(i);
				h_vec = hidden.row(k - 1);
				double O = W_i.dot(h_vec);
				hidden(k, i) = Sigmoid(O);
			}
		}

		/*feed forward y*/
		h_vec = hidden.row(NUMBER_OF_HIDDEND_LAYER - 1);
		output = V.dot(h_vec); //last hidden layer
		output = Sigmoid(output);
		log << output << endl;
		if (output > 0.5)
			output = 1;
		else if (output <= 0.5)
			output = 0;

		if (output == r)
			correct++;

	}
	cout << "Result" << (double)correct / (double)N * 100 << endl;
}

void Print(MatrixXd **W, MatrixXd first_W, VectorXd V) {
	ofstream log2("log2.txt", ios::app);
	log2 << "V matrix" << endl;
	log2 << V << endl;
	log2 << "# " << 0 << " W matrix" << endl;
	log2 << first_W << endl;
	if (NUMBER_OF_HIDDEND_LAYER > 1)
	{
		for (int i = 1; i < NUMBER_OF_HIDDEND_LAYER; i++)
		{
			log2 << "# " << i << " W matrix" << endl;
			log2 << (*W[i]) << endl;
		}
	}
	return;
}


void RBM(MatrixXd &W, double ** datum, int inputdata_cnt, int visible_cnt, int hidden_cnt) {
	/*RBM*/
	double diff_sum;
	int rbm_epoch = RBM_EPOCH;
	while (rbm_epoch--)
	{
		vector<int> rand_sequence = rand_order_gen(inputdata_cnt); //random sequence generated
		diff_sum = 0;
		cout << "# RBM EPOCH " << rbm_epoch << " " << endl;;
		MatrixXd P = MatrixXd(2, hidden_cnt);
		VectorXd H = VectorXd(hidden_cnt);
		VectorXd Vi = VectorXd(visible_cnt);
		MatrixXd W_sum = MatrixXd(hidden_cnt, visible_cnt);

		for (int i = 0; i < hidden_cnt; i++)
		{
			for (int j = 0; j < visible_cnt; j++)
			{
				W_sum(i, j) = 0;
			}
		}

		for (int t = 0; t < inputdata_cnt; t++)
		{
			VectorXd input = array2vector(datum[rand_sequence[t]]);
			P(0, 0) = 1;
			P(1, 0) = 1;
			H(0) = 1;
			Vi(0) = 1;
			for (int j = 1; j < hidden_cnt; j++)
			{
				VectorXd first_W_j(visible_cnt);
				first_W_j = W.row(j);
				double O = first_W_j.dot(input);
				P(0, j) = (1.0) / (1.0 + (exp(-O)));
			}
			for (int j = 1; j < hidden_cnt; j++)
			{
				if ((rand() % 11) < P(0, j) * 10)
					H(j) = 1;
				else
					H(j) = 0;
			}
			for (int i = 1; i < visible_cnt; i++)
			{
				VectorXd first_W_i(hidden_cnt);
				first_W_i = W.col(i);
				double O = 0;
				for (int j = 1; j< hidden_cnt; j++)
				{
					O += first_W_i(j)*P(0, j);
				}
				Vi(i) = (1.0) / (1.0 + (exp(-O)));
			}
			for (int j = 1; j < hidden_cnt; j++)
			{
				VectorXd first_W_j(hidden_cnt);
				first_W_j = W.row(j);
				double O = first_W_j.dot(Vi);
				P(1, j) = (1.0) / (1.0 + (exp(-O)));
			}
			for (int i = 1; i < hidden_cnt; i++)
			{
				for (int j = 0; j < visible_cnt; j++)
				{
					double diff = RBM_LEARNING_RATE*(input(j)*P(0, i) - Vi(j)*P(1, i));
					W_sum(i, j) += diff;
				}
			}

		}

		for (int i = 1; i < hidden_cnt; i++)
		{
			for (int j = 0; j < visible_cnt; j++)
			{
				W(i, j) += W_sum(i, j) / (double)inputdata_cnt;
			}
		}
	}
	//cout << "RBM complete!" << endl;
}

double ** Normalize(double ** data, int inputdata_cnt)
{
	double ** normal_input = (double**)malloc(sizeof(double*) * inputdata_cnt);
	for (int i = 0; i < inputdata_cnt; i++)
	{
		normal_input[i] = (double*)malloc(sizeof(double) * DIMENSION);
		memset(normal_input[i], 0, sizeof(double)* DIMENSION);
	}
	double sum[DIMENSION];
	double avg[DIMENSION];
	double sqr_sum[DIMENSION];
	double var[DIMENSION];
	memset(sum, 0, DIMENSION * sizeof(double));
	memset(sqr_sum, 0, DIMENSION * sizeof(double));
	for (int i = 0; i < inputdata_cnt; i++)
	{
		for (int j = 1; j < DIMENSION; j++)
		{
			sum[j] += data[i][j];
		}
	}
	for (int j = 1; j < DIMENSION; j++)
		avg[j] = sum[j] / (double)inputdata_cnt;
	for (int i = 0; i < inputdata_cnt; i++)
	{
		for (int j = 1; j < DIMENSION; j++)
		{
			sqr_sum[j] += pow((data[i][j] - avg[j]), 2);
		}
	}
	for (int j = 1; j < DIMENSION; j++)
		var[j] = sqr_sum[j] / (double)inputdata_cnt;
	for (int i = 0; i < inputdata_cnt; i++)
	{
		normal_input[i][0] = 1;
		for (int j = 1; j < DIMENSION; j++)
		{
			normal_input[i][j] = (data[i][j] - avg[j]) / sqrt(sqr_sum[j]);
		}
	}
	return normal_input;
}

int main(void)
{
	ifstream traingdata("./data/trn_.txt");
	ofstream log("log.txt");
	ofstream log2("log2.txt");
	stringstream stream;
	int N = 0;
	int inputdata_cnt;
	int testdata_cnt;
	char inputdata[MAX_DATA_COUNT];
	int epoch = EPOCH;
	double error_sum = 0;
	srand(time(NULL));


	double ** datum = (double**)malloc(sizeof(double*) * MAX_DATA_COUNT);
	int * lable = (int*)malloc(sizeof(int)*MAX_DATA_COUNT);
	for (int i = 0; i < MAX_DATA_COUNT; i++)
	{
		datum[i] = (double*)malloc(sizeof(double) * DIMENSION);
		memset(datum[i], 0, sizeof(double) * DIMENSION);
	}
	memset(lable, -1, sizeof(int)*MAX_DATA_COUNT);

	/*hidden layer 생성*/
	MatrixXd hidden(NUMBER_OF_HIDDEND_LAYER, NUMBER_OF_NODE);
	/*output layer 생성*/
	double output;
	/*W 생성*/
	MatrixXd first_W(NUMBER_OF_NODE, DIMENSION);
	setRandFill(first_W);
	MatrixXd * W[NUMBER_OF_HIDDEND_LAYER];
	for (int i = 1; i < NUMBER_OF_HIDDEND_LAYER; i++)
	{
		W[i] = new MatrixXd(NUMBER_OF_NODE, NUMBER_OF_NODE);
		setRandFill(W[i]);
	}
	MatrixXd first_Prev_W_diff(NUMBER_OF_NODE, DIMENSION);
	MatrixXd * Prev_W_diff[NUMBER_OF_HIDDEND_LAYER];
	for (int i = 0; i < NUMBER_OF_HIDDEND_LAYER; i++)
	{
		Prev_W_diff[i] = new MatrixXd(NUMBER_OF_NODE, NUMBER_OF_NODE);
	}
	/*V 생성*/
	VectorXd V(NUMBER_OF_NODE);
	VectorXd Prev_V_diff(NUMBER_OF_NODE);
	setRandFill(V);

	VectorXd h_vec(NUMBER_OF_NODE);
	VectorXd prev_h_vec(NUMBER_OF_NODE);

	/*Test data read*/
	ifstream tstdata("./data/tst.txt");
	double ** datum_t = (double**)malloc(sizeof(double*) * MAX_DATA_COUNT);
	int * lable_t = (int*)malloc(sizeof(int)*MAX_DATA_COUNT);
	for (int i = 0; i < MAX_DATA_COUNT; i++)
	{
		datum_t[i] = (double*)malloc(sizeof(double) * DIMENSION);
		memset(datum_t[i], 0, sizeof(double) * DIMENSION);
	}
	memset(lable_t, -1, sizeof(int)*MAX_DATA_COUNT);
	N = 0;
	while (tstdata.getline(inputdata, MAX_INPUTDATA_LEN))
	{
		stream.str(inputdata);
		datum_t[N][0] = 1; //bias
		for (int i = 1; i < DIMENSION; i++) // 
		{
			stream >> datum_t[N][i];
		}
		stream >> lable_t[N];
		stream.clear();
		N++;
	}
	testdata_cnt = N;
	tstdata.close();


	/*Taining data read*/
	while (traingdata.getline(inputdata, MAX_INPUTDATA_LEN))
	{
		stream.str(inputdata);
		datum[N][0] = 1;
		for (int i = 1; i < DIMENSION; i++) // 
		{
			stream >> datum[N][i];
		}
		stream >> lable[N];
		stream.clear();
		N++;
	}
	cout << "Training data read OK!" << endl;
	inputdata_cnt = N;
	traingdata.close();
	












	/*RBM*/
	//normalize
	double ** normal_input = Normalize(datum, inputdata_cnt);
	RBM(first_W, normal_input, inputdata_cnt, DIMENSION, NUMBER_OF_NODE);

	double ** visible_tmp = (double**)malloc(sizeof(double*) * inputdata_cnt);
	for (int i = 0; i < inputdata_cnt; i++)
	{
		visible_tmp[i] = (double*)malloc(sizeof(double) * MAX_NUMBER_OF_NODE);
		memset(visible_tmp[i], 0, sizeof(double)* MAX_NUMBER_OF_NODE);
	}
	for (int i = 0; i < inputdata_cnt; i++)
	{
		for (int j = 1; j < NUMBER_OF_NODE; j++) {
			visible_tmp[i][j] = normal_input[i][j];
		}
	}

	for (int n = 1; n < NUMBER_OF_HIDDEND_LAYER; n++)
	{
		double ** visible = (double**)malloc(sizeof(double*) * inputdata_cnt);
		for (int i = 0; i < inputdata_cnt; i++)
		{
			visible[i] = (double*)malloc(sizeof(double) * MAX_NUMBER_OF_NODE);
			memset(visible[i], 0, sizeof(double)* MAX_NUMBER_OF_NODE);
		}
		for (int i = 0; i < inputdata_cnt; i++)
		{
			visible[i][0] = 1;
			for (int j = 1; j < NUMBER_OF_NODE; j++) {
				for (int k = 0; k < DIMENSION; k++)
				{
					if (n>1)
						visible[i][j] += (*W[n])(j, k) * visible_tmp[i][k];
					else
						visible[i][j] += first_W(j, k) * visible_tmp[i][k];
				}
			}
		}
		for (int i = 0; i < inputdata_cnt; i++)
		{
			for (int j = 1; j < NUMBER_OF_NODE; j++) {
				visible_tmp[i][j] = visible[i][j];
			}
		}

		RBM(*W[n], visible, inputdata_cnt, NUMBER_OF_NODE, NUMBER_OF_NODE);
		for (int i = 0; i < inputdata_cnt; i++)
			free(visible[i]);
		free(visible);
	}
	for (int i = 0; i < inputdata_cnt; i++)
		free(normal_input[i]);
	free(normal_input);


	Print(W, first_W, V);

	/*EPOCH*/
	while (epoch--) {
		cout << "# EPOCH " << epoch << " ";
		error_sum = 0;
		vector<int> rand_sequence = rand_order_gen(inputdata_cnt); //random sequence generated
		for (int t = 0; t < inputdata_cnt; t++)
		{
			/*DATA SET*/
			VectorXd input = array2vector(datum[rand_sequence[t]]);
			double r = lable[rand_sequence[t]];

			/*feed forward h*/
			hidden(0, 0) = 1;
			for (int i = 1; i < NUMBER_OF_NODE; i++) {
				VectorXd first_W_i(DIMENSION);
				first_W_i = first_W.row(i);
				double O = first_W_i.dot(input);
				hidden(0, i) = Sigmoid(O);
			}
			for (int k = 1; k < NUMBER_OF_HIDDEND_LAYER; k++)
			{
				VectorXd W_i(NUMBER_OF_NODE);
				hidden(k, 0) = 1;
				for (int i = 1; i < NUMBER_OF_NODE; i++) {
					W_i = (*W[k]).row(i);
					h_vec = hidden.row(k - 1);
					double O = W_i.dot(h_vec);
					hidden(k, i) = Sigmoid(O);
				}
			}

			/*feed forward y*/
			h_vec = hidden.row(NUMBER_OF_HIDDEND_LAYER - 1);
			output = V.dot(h_vec); //last hidden layer
			output = Sigmoid(output);


			/*error back propagate W*/
			double DeltaN = -(r - output);
			error_sum += abs(DeltaN);
			double Delta[NUMBER_OF_HIDDEND_LAYER][NUMBER_OF_NODE];
			h_vec = hidden.row(NUMBER_OF_HIDDEND_LAYER - 1);

			for (int k = 1; k < NUMBER_OF_NODE; k++)
			{
				Delta[NUMBER_OF_HIDDEND_LAYER - 1][k] = DeltaN*V(k) * h_vec(k)*(1.0 - h_vec(k));
				//	delta[1] += abs(DeltaN*V(k) * h_vec(k)*(1.0 - h_vec(k)));
			}

			for (int n = NUMBER_OF_HIDDEND_LAYER - 2; n >= 0; n--)
			{
				for (int l = 1; l < NUMBER_OF_NODE; l++)
				{
					double sum = 0;
					VectorXd W_l(NUMBER_OF_NODE);
					W_l = (*W[n + 1]).col(l);
					for (int k = 1; k < NUMBER_OF_NODE; k++)
					{
						sum += Delta[n + 1][k] * W_l(k);
					}
					h_vec = hidden.row(n);
					Delta[n][l] = sum * h_vec(l)*(1.0 - h_vec(l));
					//delta[0]+=abs(sum * h_vec[l]);
				}
			}


			/*W first update*/
			for (int i = 1; i < NUMBER_OF_NODE; i++)
			{
				for (int j = 0; j < DIMENSION; j++)
				{
					double diff_w_i_j = Delta[0][i] * input[j];
					if (t > 0)
					{
						first_W(i, j) = first_W(i, j) - LEARNING_RATE*diff_w_i_j + MOMENTUM*first_Prev_W_diff(i, j);
						first_Prev_W_diff(i, j) = -LEARNING_RATE*diff_w_i_j;
					}
					else
					{
						first_W(i, j) = first_W(i, j) - LEARNING_RATE*diff_w_i_j;
						first_Prev_W_diff(i, j) = -LEARNING_RATE* diff_w_i_j;
					}
				}
			}

			for (int n = 1; n < NUMBER_OF_HIDDEND_LAYER; n++)
			{
				h_vec = hidden.row(n - 1);
				for (int i = 1; i < NUMBER_OF_NODE; i++)
				{
					for (int j = 0; j < NUMBER_OF_NODE; j++)
					{
						double diff_w_i_j = Delta[n][i] * h_vec(j);
						if (t > 0)
						{
							(*W[n])(i, j) = (*W[n])(i, j) - LEARNING_RATE*diff_w_i_j + MOMENTUM*(*Prev_W_diff[n])(i, j);
							(*Prev_W_diff[n])(i, j) = -LEARNING_RATE*diff_w_i_j;
						}
						else
						{
							(*W[n])(i, j) = (*W[n])(i, j) - LEARNING_RATE*diff_w_i_j;
							(*Prev_W_diff[n])(i, j) = -LEARNING_RATE*diff_w_i_j;
						}
					}
				}
			}

			/*error back propagate V*/
			h_vec = hidden.row(NUMBER_OF_HIDDEND_LAYER - 1);
			for (int i = 0; i < NUMBER_OF_NODE; i++)
			{
				if (t > 0)
				{
					V(i) = V(i) - LEARNING_RATE * DeltaN*h_vec(i) + MOMENTUM*Prev_V_diff(i);
					Prev_V_diff(i) = -LEARNING_RATE * DeltaN*h_vec(i);
				}
				else
				{
					V(i) = V(i) - LEARNING_RATE * DeltaN*h_vec(i);
					Prev_V_diff(i) = -LEARNING_RATE * DeltaN*h_vec(i);
				}
			}
		}
		cout << error_sum / (double)inputdata_cnt << endl;
	}
	Print(W, first_W, V);
	TEST(W, first_W, V, datum_t, lable_t, testdata_cnt);
	return 0;

}