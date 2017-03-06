#include <stdio.h>
#include <istream>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sstream>
#include <math.h>
#include "eigen-eigen\eigen-eigen-b9cd8366d4e8\Eigen\Dense"

using namespace std;
using Eigen::MatrixXd;

#define MAX_INPUTDATA_LEN 1000
#define MAX_DATA_COUNT 100000
#define DIMENSION 13 

double getLogPosterior(double * input, double * mean, MatrixXd V,double det, double prior)
{
	MatrixXd x_minus_mean(DIMENSION, 1); //x-m
	MatrixXd x_minus_mean_T(1, DIMENSION); // transpos of (x-m)
	MatrixXd V_T(DIMENSION, DIMENSION); // covariance matrix
	V_T = V.transpose();

	for (int i = 0; i < DIMENSION; i++)
	{
		x_minus_mean(i, 0) = input[i] - mean[i];
	}
	x_minus_mean_T = x_minus_mean.transpose();

	MatrixXd mid_value(1, 1);
	mid_value = (x_minus_mean_T*V.inverse())*x_minus_mean ; //(x-m)'s transpos X COV matrix's inverse X (x-m)
	double _mid_value = mid_value.sum();;

	double log_det = log(det); // ln(covariance's det)
	double result = -_mid_value/2 - log_det/2 + log(prior);
	return result;

}

void predict(double **datum,int *correct,int total_cnt, double * mean_P, double * mean_F, MatrixXd V_P, MatrixXd V_F, double PriorP_P, double PriorP_F, double* result, double weight)
{
	int FN = 0;
	int TP = 0;
	int FP = 0;
	int TN = 0;
	int N = 0;
	
	double V_P_det = V_P.determinant();
	double V_F_det = V_F.determinant();

	while (N < total_cnt)
	{
		double LogPosteriorP_P = getLogPosterior(datum[N], mean_P, V_P, V_P_det, PriorP_P);
		double LogPosteriorP_F = getLogPosterior(datum[N], mean_F, V_F, V_F_det, PriorP_F);

		if (LogPosteriorP_P+weight > LogPosteriorP_F)
		{
			if (correct[N] == 1)
			{
				TP++;
			}
			else
			{
				FP++;
			}
		}
		else
		{
			if (correct[N] == 0)
			{
				TN++;
			}
			else
			{
				FN++;
			}
		}
		N++;
	}
	double TPR = double(TP) / ((double)TP + (double)FN);
	double FPR = double(FP) / ((double)FP + (double)TN);
	double FNR = double(FN) / ((double)FN + (double)TP);
	double min = 10000;
	if (result != NULL)
	{
		result[0] = TPR;
		result[1] = FPR;
		result[2] = FNR;
		if (min > abs(FNR - FPR))
		{
			min = abs(FNR - FPR);
		}
		cout << "TPR : " << TPR << endl;
		cout <<"FPR : "<< FPR << endl;
		cout <<"FNR : "<< FNR << endl;

	}
	else {
		printf("True positive : %d\n", TP);
		printf("True negative : %d\n", TN);
		printf("False positive : %d\n", FP);
		printf("False negative : %d\n", FN);
		printf("%d\n", N);
	}
	
	printf("error : %f % \n", ((double)(FP + FN) / (double)N) * 100);

	
}


/*변수명의 끝 _P 는  CLASS가 1, _T 는 0임을 나타냄 */
int main(void)
{
	ifstream traingdata("./data/trn.txt");
	ifstream tstdata("./data/tst.txt");
	stringstream stream;
	int C_P_cnt = 0;
	int C_F_cnt = 0;
	int N = 0;
	int d = DIMENSION;
	double mean_P[DIMENSION];
	double mean_F[DIMENSION];
	double LogPosteriorP_P = 0;
	double LogPosteriorP_F = 0;
	double PriorP_P = 0;
	double PriorP_F = 0;

	char inputdata[MAX_DATA_COUNT];

	double ** datum = (double**)malloc(sizeof(double*) * MAX_DATA_COUNT);
	double ** tst_datum = (double**)malloc(sizeof(double*) * MAX_DATA_COUNT);
	int * lable = (int*)malloc(sizeof(int)*MAX_DATA_COUNT);
	int * correct = (int*)malloc(sizeof(int)*MAX_DATA_COUNT);
	for (int i = 0; i < MAX_DATA_COUNT; i++)
	{
		datum[i] = (double*)malloc(sizeof(double) * DIMENSION);
		memset(datum[i], 0, sizeof(double) * DIMENSION);
		tst_datum[i] = (double*)malloc(sizeof(double) * DIMENSION);
		memset(tst_datum[i], 0, sizeof(double) * DIMENSION);
	}
	memset(lable, -1, sizeof(int)*MAX_DATA_COUNT);
	memset(correct, -1, sizeof(int)*MAX_DATA_COUNT);
	memset(mean_P, 0, sizeof(double) * DIMENSION);
	memset(mean_F, 0, sizeof(double) * DIMENSION);

	while (traingdata.getline(inputdata, MAX_INPUTDATA_LEN))
	{
		stream.str(inputdata);
		if (inputdata[0] == ' ')
			continue;
		for (int i = 0; i < DIMENSION; i++) // DATA 읽어드리기
		{
			stream >> datum[N][i];
		}
		stream >> lable[N];
		stream.clear();
		if (lable[N] == 1) // p class 갯수
		{
			for (int j = 0; j < DIMENSION; j++)
			{
				mean_P[j] += datum[N][j]; //평균값 계산을 위하 더하기
			}
			C_P_cnt++;
		}
		else if (lable[N] == 0) // f class 갯수
		{
			for (int j = 0; j < DIMENSION; j++)
			{
				mean_F[j] += datum[N][j];
			}
			C_F_cnt++;
		}
		N++;
	}
	traingdata.close();
	for (int j = 0; j < DIMENSION; j++) // 각 클래스에 대한 평균 계산
	{
		mean_P[j] /= (double)C_P_cnt;
		mean_F[j] /= (double)C_F_cnt;
	}
	PriorP_P = (double)C_P_cnt / (double)N; //ln(P(Cp))
	PriorP_F = (double)C_F_cnt / (double)N; //ln(P(Cf))

	MatrixXd V_P(DIMENSION, DIMENSION);
	MatrixXd x_minus_mean_P(DIMENSION, 1);
	MatrixXd x_minus_mean_P_T(1, DIMENSION);
	MatrixXd V_F(DIMENSION, DIMENSION);
	MatrixXd x_minus_mean_F(DIMENSION, 1);
	MatrixXd x_minus_mean_F_T(1, DIMENSION);

	/*Covariance 행렬 초기화*/
	for (int i = 0; i < DIMENSION; i++)
	{
		for (int j = 0; j < DIMENSION; j++)
		{
			V_P(i, j) = 0;
			V_F(i, j) = 0;
		}
	}

	/*variance 구하기*/
	for (int i = 0; i < N; i++)
	{
		if (lable[i] == 1)
		{
			for (int j = 0; j < DIMENSION; j++)
			{
				x_minus_mean_P(j, 0) = datum[i][j] - mean_P[j];
			}
			x_minus_mean_P_T = x_minus_mean_P.transpose();
			V_P += x_minus_mean_P*x_minus_mean_P_T;
		}
		else
		{
			for (int j = 0; j < DIMENSION; j++)
			{
				x_minus_mean_F(j, 0) = datum[i][j] - mean_F[j];
			}
			x_minus_mean_F_T = x_minus_mean_F.transpose();
			V_F += x_minus_mean_F*x_minus_mean_F_T;
		}
	}
	for (int i = 0; i < DIMENSION; i++) 
	{
		for (int j = 0; j < DIMENSION; j++)
		{
			V_P(i, j) /= (double)C_P_cnt;
			V_F(i, j) /= (double)C_F_cnt;
		}
	}
	double V_P_det = V_P.determinant(); //variance matrix의 determinant 미리 구해놓기
	double V_F_det = V_F.determinant();

	/*test data*/
	N = 0;
	while (tstdata.getline(inputdata, MAX_INPUTDATA_LEN))
	{
		stream.str(inputdata);
		if (inputdata[0] == ' ')
			continue;
		for (int i = 0; i < DIMENSION; i++)
		{
			stream >> tst_datum[N][i];
		}
		stream >> correct[N];
		stream.clear();
		N++;
	}
	/*predict*/
	predict(tst_datum, correct, N, mean_P, mean_F, V_P, V_F, PriorP_P, PriorP_F,NULL,0);


	/*ROC를 위한 데이터 반복적인 predict*/
	double result[1000][3];
	for (int i = 0; i < 1000; i++)
		memset(result[i], -1, sizeof(double) * 2);
	int k = 0;
	printf("from -23 to 8 , weigth varying for getting ROC Curve variables\n");
	for (double i =-23; i <= 8; i +=0.1) //경험적인 값을 넣음
	{
		predict(tst_datum, correct, N, mean_P, mean_F, V_P, V_F, PriorP_P, PriorP_F, result[k], i);
		k++;
	}
	ofstream ROC("./rocdta.txt");
	double min = 100000;
	int min_index;
	for (int i = 0; i < 1000; i++)
	{
		if (result[1000][0] == -1)
		{
			ROC.close();
			break;
		}
			ROC << result[i][1] << "\t" << result[i][0] << endl;
			if (abs(result[i][1] - result[i][2])<min)
			{
				min = abs( result[i][1] - result[i][2]);
				min_index = i;
			}
		printf("FPR : %f , ", result[i][0]);
		printf("TPR : %f \n", result[i][1]);
	}
	ROC.close();
	cout << "EER : ";
	cout << result[min_index][1] <<"\t" <<result[min_index][2] << endl;

	return 0;

}