#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Header below added by Tulsi */
#include <stdlib.h>


/* Header below added by Qahwa for replaced CUBLAS code */
#include <cublas_v2.h>
#include <cuda_runtime.h>

int N = 3;
void mvt( double A[N][N],
           double x_1[N],
           double x_2[N],
           double y_1[N],
           double y_2[N])
{
   

    cublasStatus_t status;
    cublasHandle_t handle;
    double *d_A = 0;
    double *x1 = 0;
    double *x2 = 0;
    double *y1 = 0;
    double *y2 = 0;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 1.0;

    cublasCreate(&handle);
    cudaMalloc((void **)&d_A, N * N * sizeof(d_A[0]));
    cudaMalloc((void **)&x1,  N * sizeof(x1[0]));
    cudaMalloc((void **)&x2,  N * sizeof(x2[0]));
    cudaMalloc((void **)&y1,  N * sizeof(y1[0]));
    cudaMalloc((void **)&y2,  N * sizeof(y2[0]));

    cudaMemcpy(d_A, A,  N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x1, x_1, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x2, x_2, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y1, y_1, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y2, y_2, N * sizeof(double), cudaMemcpyHostToDevice);


    cublasDgemv(handle, CUBLAS_OP_T, N, N, &cublas_alpha, d_A, N, y1, 1, &cublas_beta, x1, 1);
    cublasDgemv(handle, CUBLAS_OP_N, N, N, &cublas_alpha, d_A, N, y2, 1, &cublas_beta, x2, 1);
    cudaMemcpy( x_1, x1, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( x_2, x2, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(x1);
    cudaFree(x2);
    cudaFree(y1);
    cudaFree(y2);
    cublasDestroy(handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 1;
  double ta[3][3] = { 31, -18, 40, 22, 6, 18, -17, -11, 26};
  double tx1[3] = { -8, -14, 5};
  double tx2[3] = { -4, 49, 25};
  double ty1[3] = { -5, -13, 2};
  double ty2[3] = { 3, -9, -1};

  mvt(ta, tx1, tx2, ty1, ty2);
  int i;
  int j;
  printf("The res x1 is \n");
  for (i = 0; i < N; i++)
  {
    
      printf(" %f ", tx1[i]);
  

    printf("\n");
  }
printf("The res x2 is \n");
  for (i = 0; i < N; i++)
  {
    
      printf(" %f ", tx2[i]);
    
    printf("\n");
  }

  printf("\n");
}

