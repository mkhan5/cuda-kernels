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
void symm(float alpha, float beta, float A[N][N], float B[N][N], float x[N],float y[N])
{
    int i,j,k;
  cublasStatus_t status;
cublasHandle_t handle;
float *d_A = 0;
float *d_x = 0;
float *d_y = 0;
float *d_B = 0;
float *d_Temp = 0;
const float d_beta = 0.0f;
cublasCreate(&handle);
cudaMalloc((void **)&d_A, N * N * sizeof(d_A[0]));
cudaMalloc((void **)&d_B,  N * N * sizeof(d_B[0]));
cudaMalloc((void **)&d_x,  N * sizeof(d_x[0]));
cudaMalloc((void **)&d_Temp,  N * sizeof(d_Temp[0]));
cudaMalloc((void **)&d_y,  N * sizeof(d_y[0]));
cudaMemcpy(d_A, A,  N * N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B,  N * N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, d_A, N, d_x, 1, &d_beta, d_Temp, 1);
cublasSgemv(handle, CUBLAS_OP_T, N, N, &alpha, d_B, N, d_x, 1, &d_beta, d_y, 1);
cublasSaxpy(handle, N, &alpha, d_Temp, 1, d_y, 1);

cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_A);
cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_Temp);
cublasDestroy(handle);
printf("The res x is \n");
  for (i = 0; i < N; i++)
  {
    //for (j = 0; j < N; j++)
    //{
      printf(" %f ", y[i]);
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
  float beta = 1;
  float a[3][3] = {1, 1, 1, 2, 5, 1, 3, 6, -3};
  float b[3][3] = {1, 1, 1, 2, 5, 1, 3, 6, -3}; //, 3, -1, -2, 5, 2, -1};
  float c[3] = {1, 2, 3 };
  float res[3];
  symm(alpha, beta, a, b,c, res);
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

