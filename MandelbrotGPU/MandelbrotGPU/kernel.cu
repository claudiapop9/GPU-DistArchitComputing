
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "config.h"

extern "C" {
#include "util.h"
}

#define CHECK(_t, _e) if (_e != cudaSuccess) { fprintf(stderr, "%s failed: %s", _t, cudaGetErrorString(_e)); goto Error;}
#define HERR(_t, _e) if (_e != cudaSuccess) { fprintf(stderr, "%s failed: f%s", _t, cudaGetErrorString(_e));}

__device__ void set_pixel(unsigned char* image, int width, int x, int y, unsigned char *c) {
	image[4 * width * y + 4 * x + 0] = c[0];
	image[4 * width * y + 4 * x + 1] = c[1];
	image[4 * width * y + 4 * x + 2] = c[2];
	image[4 * width * y + 4 * x + 3] = 255;
}

__global__ void generate_image_kernel(unsigned char* image, unsigned char* colormap, int width, int height, int maxInteration)
{
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	int row, col, iteration;
	double c_re, c_im, x, y, x_new;

	row = index / width;
	col = index % width;

	c_re = (col - width / 2.0)*4.0 / width;
	c_im = (row - height / 2.0)*4.0 / width;
	x = 0, y = 0;
	iteration = 0;

	while (x*x + y * y <= 4 && iteration < maxInteration) {
		x_new = x * x - y * y + c_re;
		y = 2 * x*y + c_im;
		x = x_new;
		iteration++;
	}
	set_pixel(image, width, col, row, &colormap[iteration * 3]);
}

int main()
{
	const int NR_BLOCKS = WIDTH * HEIGHT / THREADS_PER_BLOCK;

	printf("NR BLOCKS: %d\n", NR_BLOCKS);
	printf("NR THREADS PER BLOCKS: %d\n", THREADS_PER_BLOCK);
	
	const int colormapSize = (MAX_ITERATION + 1) * 3 * sizeof(unsigned char);
	unsigned char* colormap = (unsigned char*)malloc(colormapSize);

	const int imageSize = WIDTH * HEIGHT * 4 * sizeof(unsigned char);
	unsigned char* image = (unsigned char*)malloc(imageSize);

	init_colormap(MAX_ITERATION, colormap);

	unsigned char* d_colormap = NULL;
	unsigned char* d_image = NULL;

	CHECK("cudaSetDevice", cudaSetDevice(0));

	CHECK("cudaMalloc colormap", cudaMalloc(&d_colormap, colormapSize));
	CHECK("cudaMalloc image", cudaMalloc(&d_image, imageSize));

	cudaEvent_t start, stop;
	CHECK("cudaEventCreate start", cudaEventCreate(&start));
	CHECK("cudaEventCreate stop", cudaEventCreate(&stop));

	CHECK("CopyArrayToDevice colormap", cudaMemcpy(d_colormap, colormap, colormapSize, cudaMemcpyHostToDevice));

	char path[255];
	float times[REPEAT];

	for (int r = 0; r < REPEAT; r++) {
		printf("Repeat: %d/%d\n", r, REPEAT);

		CHECK("EventRecord start", cudaEventRecord(start, 0));

		generate_image_kernel <<<NR_BLOCKS, THREADS_PER_BLOCK >>> (d_image, d_colormap, WIDTH, HEIGHT, MAX_ITERATION);
		CHECK("kernel", cudaGetLastError());
		
		CHECK("cudaMemcpyfromDevice image", cudaMemcpy(image, d_image, imageSize, cudaMemcpyDeviceToHost));

		CHECK("cudaEventRecord stop", cudaEventRecord(stop, 0));
		CHECK("cudaEventSynchronize stop", cudaEventSynchronize(stop));

		CHECK("cudaEventElapsedTime duration", cudaEventElapsedTime(&times[r], start, stop));
		times[r] /= 1000.0;

		sprintf(path, IMAGE, "gpu", r);
		save_image(path, image, WIDTH, HEIGHT);
		progress("gpu", r, times[r]);
	}

	report("gpu", times);

Error:
	free(colormap);
	free(image);

	HERR("Free d_colormap", cudaFree(d_colormap));
	HERR("Free d_image", cudaFree(d_image));
	HERR("cudaEventDestroy start", cudaEventDestroy(start));
	HERR("cudaEventDestroy stop", cudaEventDestroy(stop));
	HERR("cudaDeviceReset", cudaDeviceReset());

	return 0;
}
