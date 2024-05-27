#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction
{
	namespace Thrust
	{
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata)
		{
			// TODO use `thrust::exclusive_scan`
			// example: for device_vectors dv_in and dv_out:

			timer().startGpuTimer();

			int* d_odata; cudaMalloc((void**)&d_odata, n * sizeof(int));
			int* d_idata; cudaMalloc((void**)&d_idata, n * sizeof(int));

			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			thrust::device_ptr<int> thrust_idata = thrust::device_ptr<int>(d_idata);
			thrust::device_ptr<int> thrust_odata = thrust::device_ptr<int>(d_odata);

			thrust::exclusive_scan(thrust_idata, thrust_idata + n, thrust_odata);

			cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);


			cudaFree(d_odata);
			cudaFree(d_idata);
			timer().endGpuTimer();
		}
	}
}
