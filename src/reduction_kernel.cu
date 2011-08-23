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
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>
//#include "sharedmem.cuh"

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

/* template <>
struct SharedMemory <int>
{
    __device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }    
};
*/

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};


#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif


/*
    This version is completely unrolled.  It uses a template parameter to achieve 
    optimal code for any (power of 2) number of threads.  This requires a switch 
    statement in the host code to handle all the different thread block sizes at 
    compile time.
*/
template <class T, unsigned int blockSize>
__global__ void
k_reduce5(T *g_idata, T *g_odata)
{

    T *sdata = SharedMemory<T>();
    // SharedMemory<T> smem;
    // T *sdata = smem.getPointer();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockSize];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
template <class T, unsigned int blockSize>
__global__ void
k_reduce5_2(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockSize < n) 
        mySum += g_idata[i+blockSize];  

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
*/
template <class T1, class T2, unsigned int blockSize>
__global__ void
k_reduce6(T1 *g_idata, T2 *g_odata, unsigned int n)
{
    T2 *sdata = SharedMemory<T2>();
    // SharedMemory<T2> smem;
    // T2 *sdata = smem.getPointer();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
      int i1 = (g_idata[i] == 0) ? 0 : 1;
      int i2 = (g_idata[i+blockSize] == 0) ? 0 : 1;
      sdata[tid] += i1 + i2;
      i += gridSize;
    } 
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
k_reduce6_2mod(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();
    int i1 = 0, i2 = 0;

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    sdata[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
      i1 = (g_idata[i] == 0) ? 0 : 1;
      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (nIsPow2 || i + blockSize < n) 
        i2 = (g_idata[i+blockSize] == 0) ? 0 : 1;
      else
        i2 = 0;
      sdata[tid] += i1 + i2;
      i += gridSize;
    } 
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] += smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] += smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] += smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] += smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] += smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] += smem[tid +  1]; EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
k_reduce6_2(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = 0;
    sdata[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] += smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] += smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] += smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] += smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] += smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] += smem[tid +  2]; EMUSYNC; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
reduce5(int size, int threads, int blocks, 
             int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(T);

	switch (threads)
	{
	case 512:
		k_reduce5<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 256:
		k_reduce5<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 128:
		k_reduce5<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 64:
		k_reduce5<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 32:
		k_reduce5<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case 16:
		k_reduce5<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case  8:
		k_reduce5<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case  4:
		k_reduce5<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case  2:
		k_reduce5<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	case  1:
		k_reduce5<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
	}
}

template <class T1, class T2>
void 
reduce6(int size, int threads, int blocks, 
             int whichKernel, T1* d_idata, T2* d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(T2);

	switch (threads)
	{
	case 512:
		k_reduce6<T1,T2, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case 256:
		k_reduce6<T1,T2, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case 128:
		k_reduce6<T1,T2, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case 64:
		k_reduce6<T1,T2,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case 32:
		k_reduce6<T1,T2,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case 16:
		k_reduce6<T1,T2,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case  8:
		k_reduce6<T1,T2,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case  4:
		k_reduce6<T1,T2,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case  2:
		k_reduce6<T1,T2,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	case  1:
		k_reduce6<T1,T2,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	}    
}
////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
reduce5_2(int size, int threads, int blocks, 
             int whichKernel, T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
	  // int smemSize = threads * sizeof(T);
    
    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

  switch (threads)
  {
  case 512:
      k_reduce5_2<T, 512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 256:
      k_reduce5_2<T, 256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 128:
      k_reduce5_2<T, 128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 64:
      k_reduce5_2<T,  64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 32:
      k_reduce5_2<T,  32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 16:
      k_reduce5_2<T,  16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  8:
      k_reduce5_2<T,   8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  4:
      k_reduce5_2<T,   4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  2:
      k_reduce5_2<T,   2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  1:
      k_reduce5_2<T,   1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  }
}


template <class T, bool nIsPow2>
void 
reduce6_2(int size, int threads, int blocks, 
             int whichKernel, T* d_idata, T* d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(T);

  switch (threads)
  {
  case 512:
      k_reduce6_2<T, 512, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 256:
      k_reduce6_2<T, 256, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 128:
      k_reduce6_2<T, 128, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 64:
      k_reduce6_2<T,  64, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 32:
      k_reduce6_2<T,  32, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 16:
      k_reduce6_2<T,  16, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  8:
      k_reduce6_2<T,   8, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  4:
      k_reduce6_2<T,   4, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  2:
      k_reduce6_2<T,   2, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  1:
      k_reduce6_2<T,   1, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  }
}

template <class T, bool nIsPow2>
void 
reduce6_2mod(int size, int threads, int blocks, 
             int whichKernel, T* d_idata, T* d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(T);

  switch (threads)
  {
  case 512:
      k_reduce6_2mod<T, 512, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 256:
      k_reduce6_2mod<T, 256, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 128:
      k_reduce6_2mod<T, 128, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 64:
      k_reduce6_2mod<T,  64, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 32:
      k_reduce6_2mod<T,  32, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case 16:
      k_reduce6_2mod<T,  16, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  8:
      k_reduce6_2mod<T,   8, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  4:
      k_reduce6_2mod<T,   4, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  2:
      k_reduce6_2mod<T,   2, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  case  1:
      k_reduce6_2mod<T,   1, nIsPow2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
  }
}
#endif // #ifndef _REDUCE_KERNEL_H_
