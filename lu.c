#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Header below added by Tulsi */
#include <stdlib.h>


/* Header below added by Qahwa for replaced CUBLAS code */
#include "cuda_runtime.h"
#include <cusolverDn.h>
#include <cublas_v2.h>

int N = 3;
void symm(double alpha, double beta, double A[N][N])
{
    double *d_A,*d_B,*d_C,*d_AT;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void **)&d_A,      N * N * sizeof(double));
    cudaMalloc((void **)&d_AT,      N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N, N, N, &cublas_alpha, d_A, N, &cublas_beta, d_B, N, d_AT, N);
    int worksize = 0;
    int *devInfo;
    cudaMalloc((void **)&devInfo, sizeof(int));
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnDgetrf_bufferSize(solver_handle, N, N, d_AT, N, &worksize);
    double *work;
    cudaMalloc((void **)&work, worksize * sizeof(double));
    int *devIpiv;
    cudaMalloc((void **)&devIpiv, N * sizeof(int));
    cusolverDnDgetrf(solver_handle, N, N, d_AT, N, work, devIpiv, devInfo);
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0)
        printf("Unsuccessful getrf execution\n\n");
    printf("\nFactorized matrix\n");
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N, N, N, &cublas_alpha, d_AT, N, &cublas_beta, d_B, N, d_A, N);
    cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_AT);
    cusolverDnDestroy(solver_handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 0;
  double a[3][3] =   { 8, 2, 1, 3, 7, 2, 2, 3, 9}; //{ 8, 3, 2, 2, 7, 3, 1, 2, 9};
  double q[3][3];
  double r[3][3];
  //double b[4][4] =  { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9}; //, 3, -1, -2, 5, 2, -1};
 // double b[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
  //double res[4][4];
  symm(alpha, beta, a);
  int i;
  int j;
  printf("The res Q is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", a[i][j]);
    }

    printf("\n");
  }


  printf("\n");
}

