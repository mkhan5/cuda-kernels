#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Header below added by Tulsi */
#include <stdlib.h>


/* Header below added by Qahwa for replaced CUBLAS code */
#include "cuda_runtime.h"
#include <cublas_v2.h>

int N = 3;
void symm(double alpha, double beta, double A[N][N],double B[N][N],double C[N][N])
{

    double *d_A,*d_B,*d_C;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 1.0;
    const double cublas_beta_for_trans = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void **)&d_A, N * N * sizeof(double));
    cudaMalloc((void **)&d_B, N * N * sizeof(double));
    cudaMalloc((void **)&d_C, N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    //Matrix Addition C = A+B
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_T, N, N, &cublas_alpha, d_A, N, &cublas_beta, d_B, N, d_C, N);
    // Transpose A = C^T
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_T, N, N, &cublas_alpha, d_C, N, &cublas_beta_for_trans, d_B, N, d_A, N);
    cudaMemcpy(C, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 0;
  double a[3][3] =  { 9, 1, 26, 12, 1, -7, 1, 4, 4};
  double b[3][3] =  {-1, 3, -3, 4, -6, 5, -5, -3, 1};
  double c[3][3];
  //double b[4][4] =  { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9}; //, 3, -1, -2, 5, 2, -1};
 // double b[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
  //double res[4][4];
  symm(alpha, beta, a,b,c);
  int i;
  int j;
  printf("The res Q is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", c[i][j]);
    }

    printf("\n");
  }

  printf("\n");
}

