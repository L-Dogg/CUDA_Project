#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <bitset>
#include <cstdlib>
#include <stdint.h>
#include <time.h>

#define SEQUENCE_LENGTH 262144 // (In bits, 64 multiples only)
#define SEQUENCES_COUNT 256
#define UINT64_CAPACITY 64

#define BLOCK_SIZE 512

const uint64_t m1 = 0x55; //(01)b
const uint64_t m2 = 0x33; //(0011)b
const uint64_t m4 = 0x0f; //(00001111)b

__host__ __device__ int bitCount(uint64_t x);
__global__ void reduce(uint64_t* input, int* output, int len);
__global__ void compareSequences(int n, const uint64_t *a, const uint64_t *b, int* distance, const int distance_idx);

void generateSequences(uint64_t** seq, int uints_required, int** reduce_output, int blocks_count);
int calculateUsingCuda(uint64_t** seq, uint64_t** dev_seq,
	int* distances, int * dev_distances,
	int** reduce_output, int** dev_reduce_output, int reduce_output_length,
	int uints_required, int blocks_count);
int countPairs(int* distance);
void calculateUsingCPU(uint64_t** seq, int uints_required, int* distances);

int main(int argc, char ** argv)
{
	int uints_required = SEQUENCE_LENGTH / UINT64_CAPACITY;
	int blocks_count = (uints_required + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int reduce_output_length = uints_required / (BLOCK_SIZE << 1);

	uint64_t **seq = nullptr, **dev_seq = nullptr;
	int *distances = nullptr, *dev_distances = nullptr;
	int **reduce_output = nullptr, **dev_reduce_output = nullptr;
	int *sum = nullptr;

	seq = (uint64_t **)malloc(SEQUENCES_COUNT * sizeof(uint64_t *));
	dev_seq = (uint64_t **)malloc(SEQUENCES_COUNT * sizeof(uint64_t *));

	reduce_output = (int **)malloc(SEQUENCES_COUNT * sizeof(int *));
	dev_reduce_output = (int **)malloc(SEQUENCES_COUNT * sizeof(int *));

	distances = (int *)malloc(SEQUENCES_COUNT * SEQUENCES_COUNT * sizeof(int));

	printf("Generating sequences.\n");
	clock_t begin = clock();
	generateSequences(seq, uints_required, reduce_output, blocks_count);
	clock_t end = clock();

	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("Generated sequences. Time elapsed: %fs.\n", elapsed_secs);
	printf("Sequences count: %d, sequence length: %d\n", SEQUENCES_COUNT, SEQUENCE_LENGTH);

	printf("Starting CUDA computation.\n");
	begin = clock();
	int ret = calculateUsingCuda(seq, dev_seq, distances, dev_distances,
		reduce_output, dev_reduce_output, reduce_output_length, uints_required, blocks_count);
	end = clock();

	if (ret != 0)
		return -1;

	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	int pairs = countPairs(distances);
	printf("Found %d pairs. Time elapsed: %fs.\n", pairs, elapsed_secs);

	cudaFree(dev_seq);
	cudaFree(dev_distances);
	cudaDeviceReset();

	memset(distances, 0, SEQUENCES_COUNT * SEQUENCES_COUNT);

	printf("Starting CPU computation.\n");
	begin = clock();
	calculateUsingCPU(seq, uints_required, distances);
	end = clock();

	pairs = countPairs(distances);
	elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("Found %d pairs. Time elapsed: %fs.\n", pairs, elapsed_secs);

	return 0;
}

void generateSequences(uint64_t** seq, int uints_required, int** reduce_output, int blocks_count)
{
	for (int i = 0; i < SEQUENCES_COUNT; i++)
	{
		srand(2137 + i);
		seq[i] = (uint64_t *)malloc(uints_required * sizeof(uint64_t));
		reduce_output[i] = (int*)malloc(sizeof(int) * blocks_count);
		for (int k = 1; k <= 63; k++)
		{
			uint64_t bit = (uint64_t)(rand() % 2) << k;
			for (int j = 0; j < uints_required; j++)
				seq[i][j] += bit;
		}

		// First half should have last bit set to 1
		if (i < SEQUENCES_COUNT / 2 && !(seq[i][0] & 0x1))
			seq[i][0] += 1;
		// Second half should have last bit set to 0
		else if (i > SEQUENCES_COUNT / 2 && (seq[i][0] & 0x1))
			seq[i][0] -= 1;		
	}

}

int countPairs(int* distances)
{
	int pairs = 0;

	for (int i = 0; i < SEQUENCES_COUNT; i++)
		for (int j = i + 1; j < SEQUENCES_COUNT; j++)
			if (distances[j + i * SEQUENCES_COUNT] == 1)
				pairs++;

	return pairs;
}

int calculateUsingCuda(uint64_t** seq, uint64_t** dev_seq,
	int* distances, int* dev_distances,
	int** reduce_output, int** dev_reduce_output, int reduce_output_length,
	int uints_required, int blocks_count)
{
	int *sums;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc(&dev_distances, SEQUENCES_COUNT * SEQUENCES_COUNT * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(dev_distances, 0, SEQUENCES_COUNT * SEQUENCES_COUNT * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	for (int i = 0; i < SEQUENCES_COUNT; i++)
	{
		// Allocate memory on the device for bit sequence
		cudaStatus = cudaMalloc(&dev_seq[i], uints_required * sizeof(uint64_t));
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Copy bit sequence from host to the device
		cudaStatus = cudaMemcpy(dev_seq[i], seq[i], uints_required * sizeof(uint64_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Allocate memory on the device for reduce output
		cudaStatus = cudaMalloc(&dev_reduce_output[i], sizeof(int) * reduce_output_length);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Init reduce output with zeros
		cudaMemset(dev_reduce_output[i], 0, sizeof(int) * reduce_output_length);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemset failed!");
			goto Error;
		}
	}

	for (int i = 0; i < SEQUENCES_COUNT; i++)
	{
		reduce << <reduce_output_length, BLOCK_SIZE >> > (dev_seq[i], dev_reduce_output[i], uints_required);
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Couldn't launch reduce: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	sums = (int *)malloc(SEQUENCES_COUNT * sizeof(int));
	memset(sums, 0, SEQUENCES_COUNT * sizeof(int));

	for (int i = 0; i < SEQUENCES_COUNT; i++)
	{
		cudaStatus = cudaMemcpy(reduce_output[i], dev_reduce_output[i], reduce_output_length * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaMemcpy failed! - reduce output\n");
			goto Error;
		}
		for (int j = 0; j < reduce_output_length; j++)
			sums[i] += reduce_output[i][j];
	}

	for (int i = 0; i < SEQUENCES_COUNT; i++)
	{
		for (int j = i + 1; j < SEQUENCES_COUNT; j++)
		{
			if (sums[i] - sums[j] == 1 || sums[i] - sums[j] == -1)
			{
				compareSequences << <blocks_count, BLOCK_SIZE >> > (uints_required, dev_seq[i], dev_seq[j], dev_distances, j + SEQUENCES_COUNT*i);

				cudaStatus = cudaGetLastError();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "Couldn't launch compareSequences: %s\n", cudaGetErrorString(cudaStatus));
					goto Error;
				}
			}
		}
	}

	cudaStatus = cudaMemcpy(distances, dev_distances, SEQUENCES_COUNT * SEQUENCES_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed! - distances");
		goto Error;
	}

	return 0;

Error:
	cudaFree(dev_seq);
	free(seq);
	cudaFree(dev_distances);
	free(distances);
	cudaDeviceReset();

	return -1;
}

void calculateUsingCPU(uint64_t** seq, int uints_required, int* distances)
{
	for (int i = 0; i < SEQUENCES_COUNT; i++)
		for (int j = i + 1; j < SEQUENCES_COUNT; j++)
			for (int k = 0; k < uints_required; k++)
				distances[j + SEQUENCES_COUNT * i] += bitCount(seq[i][k] ^ seq[j][k]);
}

__global__ void compareSequences(int n, const uint64_t *a, const uint64_t *b, int* distance, const int distance_idx)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n && a[index] != b[index] && distance[distance_idx] < 2)
	{
		atomicAdd(&distance[distance_idx], bitCount(a[index] ^ b[index]));
	}
}

/*
* Source:
*	http://stackoverflow.com/questions/109023/how-to-count-the-number-of-set-bits-in-a-32-bit-integer
*/
__host__ __device__ int bitCount(uint64_t x)
{
	x -= (x >> 1) & m1;
	x = (x & m2) + ((x >> 2) & m2);
	x = (x + (x >> 4)) & m4;
	x += x >> 8;
	x += x >> 16;
	x += x >> 32;

	return x & 0x7f;
}

__global__ void reduce(uint64_t* input, int* output, int len)
{
	//Load a segment of the input vector into shared memory
	__shared__ unsigned int partialSum[2 * BLOCK_SIZE];
	unsigned int t = threadIdx.x, start = 2 * blockIdx.x * BLOCK_SIZE;

	if (start + t < len)
		partialSum[t] = bitCount(input[start + t]);
	else
		partialSum[t] = 0;

	if (start + BLOCK_SIZE + t < len)
		partialSum[BLOCK_SIZE + t] = bitCount(input[start + BLOCK_SIZE + t]);
	else
		partialSum[BLOCK_SIZE + t] = 0;

	// Traverse the reduction tree
	for (unsigned int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}

	// Write the computed sum of the block to the output vector at the correct index
	if (t == 0)
		output[blockIdx.x] = partialSum[0];
}