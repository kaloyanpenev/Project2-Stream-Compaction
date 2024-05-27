#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction
{
	namespace CPU
	{
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		void simpleScan(int n, int* odata, const int* idata)
		{

			if (n < 1)
			{
				return;
			}
			odata[0] = 0;
			if (n < 2)
			{
				return;
			}

			odata[1] = idata[0];
			for (int i = 2; i < n; i++)
			{
				odata[i] = idata[i - 1] + odata[i - 1];
			}
		}

		/**
		 * CPU scan (prefix sum).
		 * For performance analysis, this is supposed to be a simple for loop.
		 * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
		 */
		void scan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();
			// TODO
			simpleScan(n, odata, idata);

			timer().endCpuTimer();
		}

		/**
		 * CPU stream compaction without using the scan function.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithoutScan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();
			// TODO
			int outputIdx = 0;
			for (int i = 0; i < n; i++)
			{
				if (idata[i] != 0)
				{
					odata[outputIdx] = idata[i];
					outputIdx++;
				}
			}

			timer().endCpuTimer();
			return outputIdx;
		}

		/**
		 * CPU stream compaction using scan and scatter, like the parallel version.
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int compactWithScan(int n, int* odata, const int* idata)
		{
			timer().startCpuTimer();

			int* binariesTemp = new int[n];
			int* binariesScanned = new int[n];

			// Build out temporary condition array
			for (int i = 0; i < n; i++)
			{
				binariesTemp[i] = idata[i] != 0 ? 1 : 0;
			}

			// Scan array
			simpleScan(n, binariesScanned, binariesTemp);

			// Scatter
			int sum = 0;
			for (int i = 0; i < n; i++)
			{
				if (binariesTemp[i] == 1)
				{
					odata[binariesScanned[i]] = idata[i];
					sum++;
				}
			}


			delete[] binariesTemp;
			delete[] binariesScanned;
			timer().endCpuTimer();
			return sum;
		}
	}
}
