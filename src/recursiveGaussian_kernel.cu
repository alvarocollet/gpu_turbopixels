/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
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
    Recursive Gaussian filter
*/

#ifndef _GAUSSIAN_KERNEL_H_
#define _GAUSSIAN_KERNEL_H_

#include "cutil_math.h"

#define BLOCK_DIM 16
#define CLAMP_TO_EDGE 1

// Transpose kernel (see transpose SDK sample for details)
__global__ void k_transpose(float *odata, float *idata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
    
    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void
k_simpleRecursive(float *id, float *od, int w, int h, float a)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x >= w) return;
    
    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float yp = (*id);  // previous output
    for (int y = 0; y < h; y++) {
        float xc = (*id);
        float yc = xc + a*(yp - xc);   // simple lerp between current and previous value
		*od = yc;
        id += w; od += w;    // move to next row
        yp = yc;
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    yp = (*id);
    for (int y = h-1; y >= 0; y--) {
        float xc = (*id);
        float yc = xc + a*(yp - xc);
		*od = ((*od) + yc)*0.5f;
        id -= w; od -= w;  // move to previous row
        yp = yc;
    }
}

/*
	recursive Gaussian filter

	parameters:	
	id - pointer to input data (RGBA image packed into 32-bit integers)
	od - pointer to output data 
	w  - image width
	h  - image height
	a0-a3, b1, b2, coefp, coefn - filter parameters
*/

__global__ void
k_recursiveGaussian(float *id, float *od, int w, int h, float a0, float a1, float a2, float a3, float b1, float b2, float coefp, float coefn)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (x >= w) return;
	
    id += x;    // advance pointers to correct column
    od += x;

    // forward pass
    float xp = 0.0f;  // previous input
    float yp = 0.0f;  // previous output
    float yb = 0.0f;  // previous output by 2
#if CLAMP_TO_EDGE
    xp = (*id); yb = coefp*xp; yp = yb;
#endif
    for (int y = 0; y < h; y++) {
        float xc = (*id);
        float yc = a0*xc + a1*xp - b1*yp - b2*yb;
		*od = (yc);
        id += w; od += w;    // move to next row
        xp = xc; yb = yp; yp = yc; 
    }

    // reset pointers to point to last element in column
    id -= w;
    od -= w;

    // reverse pass
    // ensures response is symmetrical
    float xn = (0.0f);
    float xa = (0.0f);
    float yn = (0.0f);
    float ya = (0.0f);
#if CLAMP_TO_EDGE
    xn = xa = (*id); yn = coefn*xn; ya = yn;
#endif
    for (int y = h-1; y >= 0; y--) {
        float xc = (*id);
        float yc = a2*xn + a3*xa - b1*yn - b2*ya;
        xa = xn; xn = xc; ya = yn; yn = yc;
		*od = (*od) + yc;
        id -= w; od -= w;  // move to previous row
    }
}

#endif // #ifndef _GAUSSIAN_KERNEL_H_
