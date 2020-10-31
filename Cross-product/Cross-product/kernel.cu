#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define CHECK(_t, _e) if (_e != cudaSuccess) { fprintf(stderr, "%s failed: %s", _t, cudaGetErrorString(_e)); goto Error;}
#define HERR(_t, _e) if (_e != cudaSuccess) { fprintf(stderr, "%s failed: %s", _t, cudaGetErrorString(_e));}

const int len = 2 * 1024 + 5;

__global__ void kernel(float *c, float *a, float *b)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < len) {
		c[i] = a[i] * b[i];
	}
}

int main()
{
	int i, ndev, bc, tc;
	cudaDeviceProp p;

	float* a = (float*)malloc(len * sizeof(float));
	float* b = (float*)malloc(len * sizeof(float));
	float* c = (float*)malloc(len * sizeof(float));

	float* da = NULL;
	float* db = NULL;
	float* dc = NULL;

	for (i = 0; i < len; i++) {
		a[i] = 0.5f;
		b[i] = 2.0f;
	}

	//print device properties
	CHECK("cudaGetDeviceCount", cudaGetDeviceCount(&ndev));
	for (i = 0; i < ndev; i++) {
		CHECK("cudaGetDeviceProperties", cudaGetDeviceProperties(&p, i));
		printf("Name: %s\n", p.name);
		printf("Compute capability: %d.%d\n", p.major, p.minor);
		printf("Max threads/block: %d\n", p.maxThreadsPerBlock);
		printf("Max block size: %d x %d x %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
		printf("Max grid size: %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
	}

	CHECK("cudaSetDevice", cudaSetDevice(0));

	CHECK("cudaMalloc da", cudaMalloc(&da, len * sizeof(float)));
	CHECK("cudaMalloc db", cudaMalloc(&db, len * sizeof(float)));
	CHECK("cudaMalloc dc", cudaMalloc(&dc, len * sizeof(float)));

	//transfer the data

	CHECK("cudaMemcpy da", cudaMemcpy(da, a, len * sizeof(float), cudaMemcpyHostToDevice));
	CHECK("cudaMemcpy db", cudaMemcpy(db, b, len * sizeof(float), cudaMemcpyHostToDevice));

	tc = 1024;
	bc = len / tc;
	if (len % tc != 0) {
		bc++;
	}

	kernel <<<bc, tc >>> (dc, da, db);
	CHECK("kernel", cudaGetLastError());

	CHECK("cudaMemcpy dc", cudaMemcpy(c, dc, len * sizeof(float), cudaMemcpyDeviceToHost));

	for (i = 0; i < len; i++) {
		if (i % 20 == 0) {
			printf("\n");
		}
		printf("% 2.0f", c[i]);
	}
	printf("\n");


Error:

	HERR("cudaFree da", cudaFree(da));
	HERR("cudaFree db", cudaFree(db));
	HERR("cudaFree dc", cudaFree(dc));
	HERR("cudaDeviceReset", cudaDeviceReset());

	return 0;
}
