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
void mm3(double A[N][N],  double B[N][N], double C[N][N], double D[N][N], double E[N][N])
{
    int i,j,k;
  cublasStatus_t status;
cublasHandle_t handle;
double *d_A = 0;
double *d_B = 0;
double *d_C = 0;
double *d_D = 0;
double *d_E = 0;
double *d_tmpAB = 0;
double *d_tmpCD = 0;
const double cublas_alpha = 1.0f;
const double cublas_beta = 0.0f;
cublasCreate(&handle);
cudaMalloc((void **)&d_A, N * N * sizeof(d_A[0]));
cudaMalloc((void **)&d_B, N * N * sizeof(d_B[0]));
cudaMalloc((void **)&d_C, N * N * sizeof(d_C[0]));
cudaMalloc((void **)&d_D, N * N * sizeof(d_D[0]));
cudaMalloc((void **)&d_E, N * N * sizeof(d_E[0]));
cudaMalloc((void **)&d_tmpAB, N * N * sizeof(d_tmpAB[0]));
cudaMalloc((void **)&d_tmpCD, N * N * sizeof(d_tmpCD[0]));
cudaMemcpy(d_A, A,  N * N * sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, N *N * sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_C, C, N *N * sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(d_D, D, N *N * sizeof(double), cudaMemcpyHostToDevice);
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &cublas_alpha, d_B, N, d_A, N, &cublas_beta, d_tmpAB, N);
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &cublas_alpha, d_D, N, d_C, N, &cublas_beta, d_tmpCD, N);
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &cublas_alpha, d_tmpCD, N, d_tmpAB, N, &cublas_beta, d_E, N);
cudaMemcpy(E, d_E, N*N*sizeof(double), cudaMemcpyDeviceToHost);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
cudaFree(d_D);
cudaFree(d_E);
cudaFree(d_tmpAB);
cudaFree(d_tmpCD);
cublasDestroy(handle);
}


int main(int argc, char **argv)
{
	 double a[3][3] = { 24, -22, -4, -11, 12, 10, -23, -18, -12};
    double b[3][3] = { 20, -25, -1, -2, -3, 14, -24, -9, -19};
    double c[3][3] = {-17, -16, 19, 8, 5, 4, 17, -13, 22};
    double d[3][3] = {-5, -8, -6, 1, -15, 23, 3, -21, 11};

  double e[3][3];
  //sol: y = (13, 31, 49)
  double alpha = 1;
  double beta = 0;
  mm3(a, b, c, d, e);
  int i;
  int j;
  printf("The res E is \n");
  for (i = 0; i < N; i++)
  {
    for (j = 0; j < N; j++)
    {
      printf(" %f ", e[i][j]);
    }

    printf("\n");
  }

  printf("\n");
}

