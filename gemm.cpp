#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <polybench.h>
#include <cublas_v2.h>
#include "gemm.h"
//#include <cublas_v2.h>

/* Header below added by Tulsi */
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#define NM 4000


/* Header below added by Tulsi for replaced CUBLAS code */
//#include <cublas_v2.h>

extern void *polybench_alloc_data(unsigned long long int n, int elt_size);
extern void polybench_free_data(void *ptr);
extern void polybench_flush_cache();
extern void polybench_prepare_instruments();
static void init_array(int ni, int nj, int nk, double *alpha, double *beta, double C[NM + 0][NM + 0], double A[NM + 0][NM + 0], double B[NM + 0][NM + 0])
{
  int i;
  int j;
  *alpha = 1;
  *beta = 0;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    C[i][j] = ((double) (((i * j) + 1) % ni)) / ni;


  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
    A[i][j] = ((double) ((i * (j + 1)) % nk)) / nk;


  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
    B[i][j] = ((double) ((i * (j + 2)) % nj)) / nj;


}

static void print_array(int ni, int nj, double C[NM + 0][NM + 0])
{
  int i;
  int j;
  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
  {
    if ((((i * ni) + j) % 20) == 0)
      fprintf(stderr, "\n");

    fprintf(stderr, "%0.2lf ", C[i][j]);
  }


  fprintf(stderr, "\nend   dump: %s\n", "C");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static void kernel_gemm(int ni, int nj, int nk, double alpha, double beta, double C[NM + 0][NM + 0], double A[NM + 0][NM + 0], double B[NM + 0][NM + 0])
{
  int i;
  int j;
  int k;
 // double status;
  cublasHandle_t handle;
  double *d_A;
  double *d_B;
  double *d_C;
  cublasCreate(&handle);
  cudaMalloc(&d_A, (ni * ni) * (sizeof(double)));
  cudaMalloc(&d_B, (ni * ni) * (sizeof(double)));
  cudaMalloc(&d_C, (ni * ni) * (sizeof(double)));
  cudaMemcpy(d_A, A, (ni * ni) * (sizeof(double)), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, (ni * ni) * (sizeof(double)), cudaMemcpyHostToDevice);
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &alpha, d_B, ni, d_A, ni, &beta, d_C, ni);
  cudaMemcpy(C, d_C, (ni * ni) * (sizeof(double)), cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv)
{
  int ni = NM;
  int nj = NM;
  int nk = NM;
  double alpha;
  double beta;
  double (*C)[NM + 0][NM + 0];
  C = (double (*)[NM + 0][NM + 0]) polybench_alloc_data((NM + 0) * (NM + 0), sizeof(double));
  ;
  double (*A)[NM + 0][NM + 0];
  A = (double (*)[NM + 0][NM + 0]) polybench_alloc_data((NM + 0) * (NM + 0), sizeof(double));
  ;
  double (*B)[NM + 0][NM + 0];
  B = (double (*)[NM + 0][NM + 0]) polybench_alloc_data((NM + 0) * (NM + 0), sizeof(double));
  ;
  init_array(ni, nj, nk, &alpha, &beta, *C, *A, *B);
  ;
 print_array(ni, nj, *C);
 kernel_gemm(ni, nj, nk, alpha, beta, *C, *A, *B);
  ;
  ;
if ((argc > 42) && (!strcmp(argv[0], "")))
    print_array(ni, nj, *C);

  free((void *) C);
  ;
  free((void *) A);
  ;
  free((void *) B);
  ;
  return 0;
}

