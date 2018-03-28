#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Header below added by Tulsi */
#include <stdlib.h>


/* Header below added by Qahwa for replaced CUBLAS code */
#include <cublas_v2.h>
#include <cuda_runtime.h>

int N = 4;
void symm(float alpha, float beta, float A[N][N], float B[N][N],  float C[N][N])
{
    int i,j,k;
cublasStatus_t status;
cublasHandle_t handle;
float *d_A = 0;
float *d_B = 0;
float *d_C = 0;
const float d_alpha = 1.0f;
cublasCreate(&handle);
cudaMalloc((void **)&d_A, N * N * sizeof(d_A[0]));
cudaMalloc((void **)&d_B, N * N * sizeof(d_B[0]));
cudaMalloc((void **)&d_C, N * N * sizeof(d_C[0]));
cudaMemcpy(d_A, A,  N * N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B,  N * N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_C, C, N *N * sizeof(float), cudaMemcpyHostToDevice);
cublasStrmm (handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, N, N, &d_alpha, d_A, N, d_B, N, d_C, N);
cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
cublasDestroy(handle);
}


int main(int argc, char **argv)
{
  int M = 3;
  int n = N;
  float alpha = 1;
  float beta = 0;
  float a[4][4] = { 1, 0, 0, 0, -17, 1, 0, 0, -8, 13, 1, 0, 7, 11, 19, 1};
  float b[4][4] =  { 3, 4, 7, 9, 5, 4, -1, 4, 8, 7, 8, 5, 4, 3, 0, 9}; //, 3, -1, -2, 5, 2, -1};
 // float b[3][3] = { -1, 3, -3, 0, -6, 5, -5, -3, 1};
  float res[4][4];
  symm(alpha, beta, a, b, res);
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

