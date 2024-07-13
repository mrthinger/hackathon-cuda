#include <stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
    int ARRAY_SIZE = 4;
    int *d_sum;
    int *d_a;
    int *d_b;

    int h_a[] = {1, 2, 3, 4};
    int h_b[] = {4, 5, 6, 7};

    cudaMalloc((void **)&d_a, sizeof(int) * ARRAY_SIZE);
    cudaMalloc((void **)&d_b, sizeof(int) * ARRAY_SIZE);
    cudaMalloc((void **)&d_sum, sizeof(int) * ARRAY_SIZE);

    cudaMemcpy(d_a, h_a, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

    add<<<1, ARRAY_SIZE>>>(d_a, d_b, d_sum);

    cudaDeviceSynchronize();
    int h_sum[ARRAY_SIZE];
    cudaMemcpy(h_sum, d_sum, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    printf("The sum is: ");
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        printf("%d ", h_sum[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_sum);
}
