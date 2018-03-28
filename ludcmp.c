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
void symm(double alpha, double beta, double A[N][N],  double b[N],double x[N])
{
    int i,j,k;
     double *d_A,*d_B,*d_AT;
     double *d_b;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMalloc((void **)&d_A, N * N * sizeof(double));
    cudaMalloc((void **)&d_AT, N * N * sizeof(double));
    cudaMalloc((void **)&d_b, N * sizeof(double));

    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
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
    cusolverDnDgetrs(solver_handle, CUBLAS_OP_N, N, N, d_AT, N, devIpiv, d_b, N, devInfo);

    //cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N, N, N, &cublas_alpha, d_AT, N, &cublas_beta, d_B, N, d_A, N);
    cudaMemcpy(x, d_b, N * sizeof(double), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_A);
    cudaFree(devInfo);
    cudaFree(work);
    cudaFree(devIpiv);
    cusolverDnDestroy(solver_handle);
printf("The res x is \n");
  for (i = 0; i < N; i++)
  {
    //for (j = 0; j < N; j++)
    //{
      printf(" %f ", x[i]);
   // }

    printf("\n");
  }

  printf("\n");
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 1;
  double a[3][3] ={ 1, -2, 3, 5, 8, -1, 2, 1, 1};
  //double b[3][3] = {1, 1, 1, 2, 5, 1, 3, 6, -3}; //, 3, -1, -2, 5, 2, -1};
  double b[3] = { 9, -5, 3};
  double res[3];
  symm(alpha, beta, a, b, res);
  int i;
  int j;
  printf("The res x is \n");
  for (i = 0; i < N; i++)
  {
   // for (j = 0; j < N; j++)
   // {
      printf(" %f ", res[i]);
   // }

    printf("\n");
  }

  printf("\n");
}

