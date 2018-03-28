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
void symm(double alpha, double beta, double A[N][N],double AT[N][N])
{

    double *d_A, *d_AT, *d_B;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void **)&d_A, N * N * sizeof(double));
    cudaMalloc((void **)&d_AT, N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    // Transpose AT = A^T  (to get row major form)
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N, N, N, &cublas_alpha, d_A, N, &cublas_beta, d_B, N, d_AT, N);
    cudaMemcpy(AT, d_AT, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_AT);
    cublasDestroy(handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 0;
  double a[3][3] =  { 9, 1, 26, 12, 1, -7, 1, 4, 4};
  double b[3][3] =  {-1, 3, -3, 4, -6, 5, -5, -3, 1};
  double at[3][3];
  //double b[4][4] =  { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9}; //, 3, -1, -2, 5, 2, -1};
 // double b[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
  //double res[4][4];
  symm(alpha, beta, a,at);
  int i;
  int j;
  printf("The res A is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", a[i][j]);
    }

    printf("\n");
  }

  printf("\n");
  printf("The res A trns is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", at[i][j]);
    }

    printf("\n");
  }

  printf("\n");
}

