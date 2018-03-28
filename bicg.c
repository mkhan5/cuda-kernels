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
void bicg( double A[N][N],
           double s[N],
           double q[N],
           double p[N],
           double r[N])
{
   
    cublasStatus_t status;
    cublasHandle_t handle;
    double *d_A = 0;
    double *d_s = 0;
    double *d_q = 0;
    double *d_p = 0;
    double *d_r = 0;
    const double cublas_alpha = 1.0;
    const double cublas_beta = 0.0;

    cublasCreate(&handle);
    cudaMalloc((void **)&d_A, N * N * sizeof(d_A[0]));
    cudaMalloc((void **)&d_s,  N * sizeof(d_s[0]));
    cudaMalloc((void **)&d_q,  N * sizeof(d_q[0]));
    cudaMalloc((void **)&d_p,  N * sizeof(d_p[0]));
    cudaMalloc((void **)&d_r,  N * sizeof(d_r[0]));

    cudaMemcpy(d_A, A,  N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, N * sizeof(double), cudaMemcpyHostToDevice);


    cublasDgemv(handle, CUBLAS_OP_T, N, N, &cublas_alpha, d_A, N, d_p, 1, &cublas_beta, d_q, 1);
    cublasDgemv(handle, CUBLAS_OP_N, N, N, &cublas_alpha, d_A, N, d_r, 1, &cublas_beta, d_s, 1);
    cudaMemcpy( q, d_q, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy( s, d_s, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_s);
    cudaFree(d_q);
    cudaFree(d_p);
    cudaFree(d_r);
    cublasDestroy(handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  double alpha = 1;
  double beta = 0;
   double a[3][3] = { 1, 2, 8, 10, 3, 9, 7, 4, 17};
    double p[3] =  {21, -13, -11};
    double r[3] = { -12, -3, 7};
	  double resS[3],resQ[3];
  bicg(a, resS, resQ, p, r);
  int i;
  int j;
  printf("The res Q is \n");
  for (i = 0; i < N; i++)
  {
    
      printf(" %f ", resQ[i]);
  

    printf("\n");
  }
printf("The res S is \n");
  for (i = 0; i < N; i++)
  {
    
      printf(" %f ", resS[i]);
    
    printf("\n");
  }

  printf("\n");
}

