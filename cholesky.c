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
    int i,j,k;

    cusolverDnHandle_t solver_handle;
    cusolverDnCreate(&solver_handle);
    double *d_A;
    cudaMalloc((void **)&d_A,      N * N * sizeof(double));
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    int worksize = 0;
    int *devInfo;
    cudaMalloc((void **)&devInfo, sizeof(int));
    cusolverDnDpotrf_bufferSize(solver_handle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, &worksize);
    double *work;
    cudaMalloc((void **)&work, worksize * sizeof(double));
    cusolverDnDpotrf(solver_handle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, work, worksize, devInfo);
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (devInfo_h != 0)
        printf("Unsuccessful potrf execution\n\n");
    printf("\nFactorized matrix\n");
    cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cusolverDnDestroy(solver_handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 0;
  double res[3][3] =  { 25, 15, -5, 15, 18, 0, -5, 0,11};
  //double b[4][4] =  { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9}; //, 3, -1, -2, 5, 2, -1};
 // double b[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
  //double res[4][4];
  symm(alpha, beta, res);
  int i;
  int j;
  printf("The res x is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", res[i][j]);
    }

    printf("\n");
  }

  printf("\n");
}

