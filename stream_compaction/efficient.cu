#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>
#define blockSize 256
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define CONFLICT_FREE_OFFSET(n) \
	((n) >> LOG_NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

namespace StreamCompaction
{
	namespace Efficient
	{
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		// TODO: Solve bank conflicts
		__global__ void SelfMadeEfficientScan(int* idata, int* odata, int n, int* sumArr, unsigned long long* time)
		{
			unsigned long long cl = clock();
			extern __shared__ int temp[];
			int thid = threadIdx.x;


			if (thid >= (n >> 1))
			{
				return;
			}

			temp[thid * 2] = idata[(blockIdx.x * blockDim.x + thid) * 2];
			temp[thid * 2 + 1] = idata[(blockIdx.x * blockDim.x + thid) * 2 + 1];

			// Upsweep - d manages the thread occupation, stride manages the binary tree creation
			int stride = 1;
			for (int d = n >> 1; d > 0; d >>= 1)
			{
				__syncthreads();
				if (thid < d)
				{
					int rhs = stride * (2 * thid + 2) - 1;
					int lhs = stride * (2 * thid + 1) - 1;
					temp[rhs] += temp[lhs];
				}
				stride <<= 1;
			}

			if (thid == 0) // 0 for exclusive scan
			{
				if (sumArr)
				{
					sumArr[blockIdx.x] = temp[n - 1];
				}
				temp[n - 1] = 0;
			}

			// Downsweep
			for (int i = 1; i < n; i <<= 1)
			{
				// stride is == n here, so half it now
				stride >>= 1;
				__syncthreads();
				if (thid < i)
				{
					int rhs = stride * (2 * thid + 2) - 1;
					int lhs = stride * (2 * thid + 1) - 1;
					int rhsCopy = temp[rhs];
					temp[rhs] += temp[lhs];
					temp[lhs] = rhsCopy;
				}
			}

			odata[(blockIdx.x * blockDim.x + thid) * 2] = temp[thid * 2];
			odata[(blockIdx.x * blockDim.x + thid) * 2 + 1] = temp[thid * 2 + 1];
			*time = clock() - cl;
		}

		void efficientScan(int N, int* odata, const int* idata)
		{
			// Handle arrays of arbitrary size - pad with zeroes till next power of 2
			const int nextPowTwo = ilog2ceil(N);
			const int numZeroes = pow(2, nextPowTwo) - N;

			const int n = N + numZeroes;

			// gridSize of scan => = scan<gridSizeL1, blockSize>(d_idata, out_dSumsL1)
			// increases if array of N can't be scanned with a single block
			const int gridSizeL1 = 1 + (n - 1) / (blockSize * 2);
			// gridsize for sums of scans => scan<gridSizeL2, gridSizeL1>(in_dSumsL1, out_dIncrL1)...
			// increases if the sums of "gridSizeL1" blocks can't be summed in a single block, i.e. gridSizeL1 - 1 > blockSize*2
			const int GridSizeL2 = 1 + (gridSizeL1 - 1) / (blockSize * 2);

			const int nextPowTwoL2 = ilog2ceil(gridSizeL1);
			const int numZeroesL2 = pow(2, nextPowTwoL2) - gridSizeL1;

			const int gridSizeL2 = GridSizeL2 + numZeroesL2;

			// gridSize for the scan of sums of scans => scan<gridSizeL3, gridSizeL2>(in_dIncrL1, out_dSumsL2)...
			const int gridSizeL3 = 1 + (gridSizeL2 - 1) / (blockSize * 2);


			// TEMP DELETE
			const int gridSize = (n + blockSize - 1) / blockSize;
			// Setup arrays
			int* d_odata;
			cudaMalloc((void**)&d_odata, n * sizeof(int));
			int* d_idata;
			cudaMalloc((void**)&d_idata, n * sizeof(int));

			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			// Setup cycle counter
			unsigned long long time;
			unsigned long long* d_time;
			cudaMalloc((void**)&d_time, sizeof(unsigned long long));

			// Setup sum (for arrays > maxblockSize)
			int* d_singleSum;
			cudaMalloc((void**)&d_singleSum, sizeof(int));
			checkCUDAError("cudaMalloc dev_posXX failed!");

			int* d_SumsL1;
			int* d_IncrL1;
			int* d_IncrL2;
			int* d_SumsL2;
			cudaMalloc((void**)&d_SumsL1, gridSizeL1 * sizeof(int));

			if (gridSizeL1 > 1)
			{
				cudaMalloc((void**)&d_IncrL1, gridSizeL1 * sizeof(int));
				cudaMalloc((void**)&d_SumsL2, gridSizeL2 * sizeof(int));
			}
			if (gridSizeL2 > 1)
			{
				cudaMalloc((void**)&d_IncrL2, gridSizeL2 * sizeof(int));
			}



			if (gridSizeL1 == 1)
			{
				SelfMadeEfficientScan << < gridSizeL1 /* = 1 */, n, n * sizeof(int) >> > (d_idata, d_odata, n, d_SumsL1, d_time);
			}
			else if (gridSizeL2 == 1)
			{
				SelfMadeEfficientScan << < gridSizeL1, blockSize, blockSize * sizeof(int) * 2 >> > (d_idata, d_odata, blockSize * 2, d_SumsL1, d_time);

				checkCUDAError("cudaMalloc dev_posCPYPRIOR failed!");
				SelfMadeEfficientScan << < gridSizeL2 /* = 1 */, blockSize, blockSize * sizeof(int) * 2 >> > (d_SumsL1, d_IncrL1, blockSize * 2, d_SumsL2, d_time);
				checkCUDAError("cudaMalloc dev_posCPYPRIOR failed!");
				Common::kernAdd << <gridSizeL1, blockSize >> > (d_odata, d_IncrL1);
				cudaDeviceSynchronize();
				checkCUDAError("cudaMalloc dev_posCPYPRIOR failed!");
			}
			else
			{
				// TODO: Expand to N levels
				SelfMadeEfficientScan << < gridSizeL1, blockSize, blockSize * sizeof(int) * 2 >> > (d_idata, d_odata, blockSize * 2, d_SumsL1, d_time);
				SelfMadeEfficientScan << < gridSizeL2, blockSize, blockSize * sizeof(int) * 2 >> > (d_SumsL1, d_IncrL1, blockSize * 2, d_SumsL2, d_time);
				SelfMadeEfficientScan << < gridSizeL3, blockSize, blockSize * sizeof(int) * 2 >> > (d_SumsL2, d_IncrL2, blockSize * 2, nullptr, d_time);

				Common::kernAdd << <gridSizeL2, blockSize >> > (d_IncrL1, d_IncrL2);
				Common::kernAdd << <gridSizeL1, blockSize >> > (d_odata, d_IncrL1);

				checkCUDAError("cudaMalloc dev_posCPYPRIOR failed!");
			}

			checkCUDAError("cudaMalloc dev_posCPYPRIOR failed!");

			cudaMemcpy(&time, d_time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMalloc dev_posCPY failed!");
			cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

			std::cout << "Efficient Scan Cycles: " << (time - 14) / 32 << std::endl;

			cudaFree(d_SumsL1);
			if (gridSizeL1 > 1)
			{
				cudaFree(d_IncrL1);
			}
			if (gridSizeL2 > 1)
			{
				cudaFree(d_SumsL2);
				cudaFree(d_IncrL2);
			}

			cudaFree(d_singleSum);
			checkCUDAError("cudaMalloc dev_posSUM failed!");
			cudaFree(d_time);
			checkCUDAError("cudaMalloc dev_posTIME failed!");
			cudaFree(d_odata);
			checkCUDAError("cudaMalloc dev_posODATA failed!");
			cudaFree(d_idata);
			checkCUDAError("cudaMalloc dev_posIDATA failed!");
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata)
		{
			timer().startGpuTimer();

			efficientScan(n, odata, idata);

			timer().endGpuTimer();
		}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int* odata, const int* idata)
		{
			timer().startGpuTimer();
			// TODO
			timer().endGpuTimer();
			return -1;
		}
	}
}
