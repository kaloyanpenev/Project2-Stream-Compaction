#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream>
#define blockSize 128

namespace StreamCompaction
{
	namespace Naive
	{
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__

		__global__ void naiveScan(int* idata, int* odata, int n, unsigned long long* time)
		{
			unsigned long long cl = clock();
			extern __shared__ int temp[];
			int thid = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (thid >= n)
			{
				return;
			}

			int pout = 0;
			int pin = 1;

			temp[pout * n + thid] = thid > 0 ? idata[thid - 1] : 0;

			__syncthreads();

			for (int offset = 1; offset < n; offset = offset * 2)
			{
				pout = 1 - pout;
				pin = 1 - pout;

				if (thid >= offset)
				{
					temp[pout * n + thid] = temp[pin * n + thid - offset] + temp[pin * n + thid];
				}
				else
				{
					temp[pout * n + thid] = temp[pin * n + thid];
				}
				__syncthreads();
			}
			odata[thid] = temp[pout * n + thid];

			*time = clock() - cl;

		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata)
		{
			timer().startGpuTimer();

			int gridSize = (n + blockSize - 1) / blockSize;

			int* d_odata; cudaMalloc((void**)&d_odata, n * sizeof(int));
			int* d_idata; cudaMalloc((void**)&d_idata, n * sizeof(int));

			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			unsigned long long time;
			unsigned long long* d_time;

			cudaMalloc(&d_time, sizeof(unsigned long long));

			naiveScan << < gridSize, blockSize, n * 2 * sizeof(int) >> > (d_idata, d_odata, n, d_time);

			cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			timer().endGpuTimer();

			std::cout << "Time: " << (time - 14) / 32 << std::endl;

			cudaFree(d_odata);
			cudaFree(d_idata);
			cudaFree(d_time);
		}
	}
}
