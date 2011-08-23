/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

/*
    Parallel reduction

    This sample shows how to perform a reduction operation on an array of values
    to produce a single value.

    Reductions are a very common computation in parallel algorithms.  Any time
    an array of values needs to be reduced to a single value using a binary 
    associative operator, a reduction can be used.  Example applications include
    statistics computaions such as mean and standard deviation, and image 
    processing applications such as finding the total luminance of an
    image.

    This code performs sum reductions, but any associative operator such as
    min() or max() could also be used.

    It assumes the input size is a power of 2.

    COMMAND LINE ARGUMENTS

    "--shmoo":         Test performance for 1 to 32M elements with each of the 7 different kernels
    "--n=<N>":         Specify the number of elements to reduce (default 1048576)
    "--threads=<N>":   Specify the number of threads per block (default 128)
    "--kernel=<N>":    Specify which kernel to run (0-6, default 6)
    "--maxblocks=<N>": Specify the maximum number of thread blocks to launch (kernel 6 only, default 64)
    "--cpufinal":      Read back the per-block results and do final sum of block sums on CPU (default false)
    "--cputhresh=<N>": The threshold of number of blocks sums below which to perform a CPU final reduction (default 1)
    
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

#include "reduction_kernel.cu"


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? n : maxThreads;
        blocks = n / threads;
    }
    else
    {
        if (n == 1) 
            threads = 1;
        else
            threads = (n < maxThreads*2) ? n / 2 : maxThreads;
        blocks = n / (threads * 2);

        if (whichKernel == 6)
            blocks = min(maxBlocks, blocks);
    }
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and 
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
template <class T>
T benchmarkReduce(int  n, 
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  int  whichKernel, 
                  T* d_idata, 
                  T* d_odata)
{
    T gpu_result = 0;
	gpu_result = 0;

	// execute the kernel
  reduce6_2mod<T, true>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);
  //reduce6<T, T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

	//reduce6_2<T, false>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);
	//reduce6_2<T, true>(n, numThreads, numBlocks, whichKernel, d_odata, d_odata);

	// check if kernel execution generated an error
	cutilCheckMsg("Kernel execution failed");

	// sum partial block sums on GPU
	int s=numBlocks;
	int kernel = (whichKernel == 6) ? 5 : whichKernel;
	while(s > 1) 
	{
		int threads = 0, blocks = 0;
		getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

	  // reduce5<T>(s, threads, blocks, kernel, d_odata, d_odata);
	  //reduce5_2<T>(s, threads, blocks, kernel, d_odata, d_odata);
	  reduce6_2<T, true>(s, threads, blocks, whichKernel, d_odata, d_odata);

		if (kernel < 3)
			s = s / threads;
		else
			s = s / (threads*2);
	}

	cudaThreadSynchronize();

    // copy final sum from device to host
    cutilSafeCallNoSync( cudaMemcpy( &gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost) );
    return gpu_result;
}

int do_reduction_count(int* in, int* out, int size)
{
	int maxThreads = 128;  // number of threads per block
    int whichKernel = 6;
    int maxBlocks = 128;
	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

	int gpu_result = benchmarkReduce<int>(size, numThreads, numBlocks, maxThreads, maxBlocks,
                                        whichKernel, in, out);
	
	return gpu_result;
}

