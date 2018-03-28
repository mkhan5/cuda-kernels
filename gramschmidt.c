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
void symm(double alpha, double beta, double A[N][N],double Q[N][N],double R[N][N])
{
    int i,j,k;

    double *d_A,*d_AT,*d_Q,*d_B;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMalloc((void **)&d_A,      N * N * sizeof(double));
    cudaMalloc((void **)&d_Q,      N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_AT,      N * N * sizeof(double));
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N, N, N, &cublas_alpha, d_A, N, &cublas_beta, d_B, N, d_AT, N);
    int worksize = 0;
    int *devInfo;
    cudaMalloc((void **)&devInfo, sizeof(int));
    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    cusolverDnDgeqrf_bufferSize(solver_handle, N, N, d_AT, N, &worksize);
    double *work;
    cudaMalloc((void **)&work, worksize * sizeof(double));
    double *TAU;
    cudaMalloc((void **)&TAU, N * sizeof(double));
    cusolverDnDgeqrf(solver_handle, N, N, d_AT, N, TAU, work, worksize, devInfo);
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0)
        printf("Unsuccessful geqrf execution\n\n");
    double *q_h,*r_h;
    q_h = (double *)malloc( N*N*sizeof( double ));
   // r_h = (double *)malloc( N*N*sizeof( double ));
    printf("\nFactorized matrix\n");
    cublasDgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N, N, N, &cublas_alpha, d_AT, N, &cublas_beta, d_B, N, d_A, N);
    cudaMemcpy(R, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            q_h[i*N+j] = 0.0;
            if (i==j)
                q_h[i*N+j] = 1.0;
        }
    }

    cudaMemcpy(d_Q, q_h, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cusolverDnDormqr(solver_handle,CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, N, N, d_AT, N, TAU, d_Q, N, work, worksize, devInfo);

    devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0)
        printf("Unsuccessful ormqr execution\n\n");
    cudaMemcpy(Q, d_Q, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_AT);
    cudaFree(d_Q);
    cudaFree(TAU);
    cudaFree(work);
    cudaFree(devInfo);
    cusolverDnDestroy(solver_handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 0;
  double a[3][3] =  { 9, 0, 26, 12, 0, -7, 0, 4, 4};
  double q[3][3];
  double r[3][3];
  //double b[4][4] =  { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9}; //, 3, -1, -2, 5, 2, -1};
 // double b[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
  //double res[4][4];
  symm(alpha, beta, a,q,r);
  int i;
  int j;
  printf("The res Q is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", q[i][j]);
    }

    printf("\n");
  }

  printf("The res R is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", r[i][j]);
    }

    printf("\n");
  }

  printf("\n");
}

