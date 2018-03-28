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
void symm(float alpha, float beta, float A[N][N], float B[N], float C[N])
{
    int i,j,k;
  cublasStatus_t status;
cublasHandle_t handle;
float *d_A = 0;
float *d_x = 0;
float *d_C = 0;
cublasCreate(&handle);
cudaMalloc((void **)&d_A, N * N * sizeof(d_A[0]));
cudaMalloc((void **)&d_x,  N * sizeof(d_x[0]));
cudaMalloc((void **)&d_C,  N * sizeof(d_C[0]));
cudaMemcpy(d_A, A,  N * N * sizeof(float), cudaMemcpyHostToDevice);

cudaMemcpy(d_x, B, N * sizeof(float), cudaMemcpyHostToDevice);
cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, d_A, N, d_x, 1, &beta, d_C, 1);

cudaMemcpy(C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_A);
cudaFree(d_x);
cublasDestroy(handle);
printf("The res x is \n");
  for (i = 0; i < N; i++)
  {
    //for (j = 0; j < N; j++)
    //{
      printf(" %f ", C[i]);
   // }

    printf("\n");
  }

  printf("\n");
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  float alpha = 1;
  float beta = 0;
  float a[3][3] = {1, 1, 1, 2, 5, 1, 3, 6, -3};
  float b[3] = {1, 2, 1 }; //, 3, -1, -2, 5, 2, -1};
  float res[3];
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

