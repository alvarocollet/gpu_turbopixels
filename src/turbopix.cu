/*
 * turbopix.cu
 *
 *  Created on: Aug 20, 2011
 *      Author: unknown 
 * Modified by: Alvaro Collet (acollet@cs.cmu.edu)
 *
 *  Copyright (c) 2011, Carnegie Mellon University.
 *  All rights reserved.
 *
 * Software License Agreement (BSD License)
 *
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <turbopix.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h> 
#include <cutil.h>
#include <assert.h>

// #include "bmp_io.h"

extern void gaussianFilter(float *d_src, float *d_dest, float *d_temp, int width, int height, float sigma, int order);
extern int do_reduction_count(int* d_idata, int* d_odata, int n);

#define STD_CONFIG dim3((img_width+15)>>4,(img_height+15)>>4),dim3(16,16)

// Device global memory
uchar4* d_uchar4_temp1;
uchar1* d_uchar1_temp1;
float* d_float_temp1;
float* d_float_temp2;
int* d_int_temp1;
float* d_speed;
float* d_speed_dx;
float* d_speed_dy;
cudaArray* d_float_array;
cudaArray* d_int2_array;
cudaArray* d_int_array;
float2* d_seed_locations;
int2* d_indirection_map;
int* d_assignment_map;


// Tex references
texture<uchar4, 1, cudaReadModeNormalizedFloat> texref_uchar4_1d;
texture<uchar1, 1, cudaReadModeNormalizedFloat> texref_uchar_1d;
texture<int2, 2, cudaReadModeElementType> texref_int2_2d;
texture<float, 2, cudaReadModeElementType> texref_float_2d;
texture<int, 2, cudaReadModeElementType> texref_int_2d;

// Device constant memory
__device__ __constant__ float2 d_seed_coords_const[N_SUPERPIXELS];

// Host globals
//unsigned long* h_input_image;
uint64_t* h_input_image;
unsigned long img_width;
long img_height;
long img_pitch;
int img_pixels;
int img_pixels_pow2;
int img_realpixels;

//
// Device Code
//

// Converts a 32-bit RGB image to normalized float grayscale
// texref_uchar4_1d : input image
__global__ void k_rgb2grayf(float* out, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockDim.x*blockIdx.x;
	int my_y = threadIdx.y + blockDim.y*blockIdx.y;

	if (my_x >= width || my_y >= height) return;

	const float4 transf = make_float4(0.2989, 0.5870, 0.1140, 0);
	float4 pixel = tex1Dfetch(texref_uchar4_1d, my_y*width+my_x);
     
	out[my_y*pitch+my_x] = transf.x*pixel.x + transf.y*pixel.y + transf.z*pixel.z;
}





// Converts an unsigned char grayscale image to normalized float grayscale
// texref_uchar_1d : input image
__global__ void k_gray2grayf(float* out, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockDim.x*blockIdx.x;
	int my_y = threadIdx.y + blockDim.y*blockIdx.y;

	if (my_x >= width || my_y >= height) return;

	float1 pixel = tex1Dfetch(texref_uchar_1d, my_y*width+my_x);
	
	out[my_y*pitch+my_x] = pixel.x; 
}

// Computes gradient and stores its magnitude
// texref_float_2d : input
__global__ void k_gradient_mag(float* out, int width, int height, int pitch)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;

	if (x >= width || y >= height) return;

	int out_idx = y*pitch+x;

	float gx = 0.5f * (tex2D(texref_float_2d, x+1, y) - tex2D(texref_float_2d, x-1, y));
	float gy = 0.5f * (tex2D(texref_float_2d, x, y+1) - tex2D(texref_float_2d, x, y-1));

	out[out_idx] = sqrtf(gx*gx + gy*gy);
}

// Computes gradient and stores its x/y components
// texref_float_2d : input
__global__ void k_gradient_xy(float* out_x, float* out_y, int width, int height, int pitch)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;

	if (x >= width || y >= height) return;

	int out_idx = y*pitch+x;

	float gx = 0.5f * (tex2D(texref_float_2d, x+1, y) - tex2D(texref_float_2d, x-1, y));
	float gy = 0.5f * (tex2D(texref_float_2d, x, y+1) - tex2D(texref_float_2d, x, y-1));

	out_x[out_idx] = gx;
	out_y[out_idx] = gy;
}

// Given a grid of rows*cols cells, each cell of size x_dist*y_dist, place a seed in each one.
// The seed is moved within a 2rad*2rad box around the center point of each cell such that
// it lies at a point of maximum speed.
// texref_float_2d : input speed map
__global__ void k_place_seeds(float2* out, int x_dist, int y_dist, int rad, int rows, int cols)
{
	int2 coord;
	float max = 0.0f;

	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;

	if (my_x >= cols || my_y >= rows)
		return;

	int cen_x = my_x * x_dist + (x_dist>>1);
	int cen_y = my_y * y_dist + (y_dist>>1);

	for (int y = -rad; y < rad; y++)
	{
		for (int x = -rad; x < rad; x++)
		{
			float val = tex2D(texref_float_2d, cen_x + x, cen_y + y);
			if (val > max)
			{
				max = val;
				coord = make_int2(cen_x+x, cen_y+y);
			}
		}
	}

	out[my_x + my_y*cols] = make_float2((float)coord.x, (float)coord.y);
}

// Initialize psi using the seed locations, by computing a distance transform. Each
// output pixel will contain the distance to the closest seed.
// d_seed_coords_const : seed locations stored in constant memory
__global__ void k_init_psi(float* out_psi, int* out_id, int n_seeds, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;
	if (my_x >= width || my_y >= height)
		return;
		
	float my_xf = (float)my_x;
	float my_yf = (float)my_y;
	float min_dist2 = 1e9;
	int id = 0;
	
	for (int i = 0; i < n_seeds; i++)
	{
		float dx = my_xf - d_seed_coords_const[i].x;
		float dy = my_yf - d_seed_coords_const[i].y;
		float dist2 = dx*dx + dy*dy;
		min_dist2 = fminf(dist2, min_dist2);
		
		if (dist2 <= 1.0f)
		{
			id = i+1;
		}
	}
	
	int idx = my_y*pitch + my_x;
	out_psi[idx] = sqrtf(min_dist2) - 1.0f;
	out_id[idx] = id;
}

// Given an image gradient magnitude map and a smoothed version of it, compute
// the speed map.
__global__ void k_calc_speed_based_on_gradient (float* grad, float* smooth_grad, float rsigma, float* out, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (my_x >= width || my_y >= height)
		return;
		
	int idx = my_y*pitch + my_x;
	
	float mag = grad[idx];
	float smooth_mag = smooth_grad[idx] * rsigma;
	
    float normGradMag = (mag / (0.1f + smooth_mag));
    out[idx] = expf(-normGradMag);
}

// Perform initial smoothing of the image. This kernel smoothes by one timestep.
__global__ void k_smooth_image (float* out, float timestep, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (my_x >= width || my_y >= height)
		return;
	
	int idx = my_y*pitch+my_x;
	
	// Read surrounding values
	float left = tex2D(texref_float_2d, my_x-1, my_y);
	float right = tex2D(texref_float_2d, my_x+1, my_y);
	float up = tex2D(texref_float_2d, my_x, my_y+1);
	float down = tex2D(texref_float_2d, my_x, my_y-1);
	float cen = tex2D(texref_float_2d, my_x, my_y);
	float downleft = tex2D(texref_float_2d, my_x-1, my_y-1);
	float downright = tex2D(texref_float_2d, my_x+1, my_y-1);
	float upleft = tex2D(texref_float_2d, my_x-1, my_y+1);
	float upright = tex2D(texref_float_2d, my_x+1, my_y+1);
	
	// Central differences
	float dx = (right - left) * 0.5f;
	float dy = (up - down) * 0.5f;
	float dxx = right - 2.0f*cen + left;
	float dyy = up - 2.0f*cen + down;
	float dxy = (upright + downleft - upleft - downright) * 0.25f;

	// Curvature calculation
	float dx_2 = dx*dx;
	float dy_2 = dy*dy;
	float curv = -(dxx*(dy_2) - 2.0f*dx*dy*dxy + dyy*(dx_2)) / (1e-16f + dx_2 + dy_2);
	
	// Update
	out[idx] = cen - timestep*curv;
}

// Evolve psi one timestep.
__global__ void k_evolve_psi (float* out, int* ids, float* speed, float* speed_dx, float* speed_dy, float timestep, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (my_x >= width || my_y >= height)
		return;
	
	int idx = my_y*pitch+my_x;
	int2 closest_point = tex2D(texref_int2_2d, my_x, my_y);
	int speed_idx = closest_point.y*pitch + closest_point.x;
	
	// Read surrounding psi values
	float left = tex2D(texref_float_2d, my_x-1, my_y);
	float right = tex2D(texref_float_2d, my_x+1, my_y);
	float up = tex2D(texref_float_2d, my_x, my_y+1);
	float down = tex2D(texref_float_2d, my_x, my_y-1);
	float cen = tex2D(texref_float_2d, my_x, my_y);
	float downleft = tex2D(texref_float_2d, my_x-1, my_y-1);
	float downright = tex2D(texref_float_2d, my_x+1, my_y-1);
	float upleft = tex2D(texref_float_2d, my_x-1, my_y+1);
	float upright = tex2D(texref_float_2d, my_x+1, my_y+1);
	
	int id_cen = tex2D(texref_int_2d, my_x, my_y);
	int id_right = tex2D(texref_int_2d, my_x+1, my_y);
	int id_up = tex2D(texref_int_2d, my_x, my_y+1);
	int id_across = tex2D(texref_int_2d, my_x+1, my_y+1);
	bool on_skel = (id_cen && (id_right || id_up || id_across));

	
	// Upwind derivatives
	
	float dx_plus = right - cen;
	float dx_minus = cen - left;
	float dy_plus = up - cen;
	float dy_minus = cen - down;

	float dx_p_max = fmaxf(0.0f, dx_plus);
	float dx_p_min = fminf(0.0f, dx_plus);
	float dx_m_max = fmaxf(0.0f, dx_minus);
	float dx_m_min = fminf(0.0f, dx_minus);
	float dy_p_max = fmaxf(0.0f, dy_plus);
	float dy_p_min = fminf(0.0f, dy_plus);
	float dy_m_max = fmaxf(0.0f, dy_minus);
	float dy_m_min = fminf(0.0f, dy_minus);
	
	float grad_plus = dx_m_max*dx_m_max + dx_p_min*dx_p_min + dy_m_max*dy_m_max + dy_p_min*dy_p_min;
	float grad_minus = dx_p_max*dx_p_max + dx_m_min*dx_m_min + dy_p_max*dy_p_max + dy_m_min*dy_m_min;
	grad_minus = sqrtf(grad_minus);
	grad_plus = sqrtf(grad_plus);
	
	/*
	float dx_1 = cen - fminf(right, left);
	float dy_1 = cen - fminf(up, down);
	float dx_plus = fmaxf(dx_1, 0.0f);
	float dx_minus = fminf(dx_1, 0.0f);
	float dy_plus = fmaxf(dy_1, 0.0f);
	float dy_minus = fminf(dy_1, 0.0f);
	float grad_plus = sqrtf(dx_plus*dx_plus + dy_plus*dy_plus);
	float grad_minus = sqrtf(dx_minus*dx_minus + dy_minus*dy_minus);*/

	
	// Central differences
	float dx = (right - left) * 0.5f;
	float dy = (up - down) * 0.5f;
	float dxx = right - 2.0f*cen + left;
	float dyy = up - 2.0f*cen + down;
	float dxy = (upright + downleft - upleft - downright) * 0.25f;

	// Curvature calculation
	float dx_2 = dx*dx;
	float dy_2 = dy*dy;
	float mag = sqrtf(dx_2+dy_2);
	float curv = (dxx*dy_2 - 2.0f*dx*dy*dxy + dyy*dx_2) / (mag*(dx_2+dy_2)+1e-16f);
	curv = fmaxf(-1.0f, fminf(1.0f, curv));

	// Doublet term
	float doublet = (dx*speed_dx[speed_idx] + dy*speed_dy[speed_idx]) / (mag+1e-16f);
	doublet = fmaxf(0.0f, doublet);

	// Calculate speed at this pixel
	float final_speed = speed[speed_idx]*(1.0f - 0.3f*curv) - doublet;
	final_speed = fminf(1.0f, fmaxf(-1.0f, final_speed));
	
	// Determine final delta_psi based on sign of speed
	float final_grad = fmaxf(0.0f, final_speed)*grad_plus + fminf(0.0f, final_speed)*grad_minus;
	
	// Update psi
	float new_psi = cen - timestep*final_grad;
	out[idx] = on_skel? cen : new_psi;
	
	// Update id
	int new_id = (cen > 0.0f && new_psi <= 0.0f) ? ids[speed_idx] : id_cen;
	ids[idx] = new_id;
}

// Initialize the jump flood algorithm for computing feature distance transform.
// Feature pixels are defined to be those that have psi<=0 (pixels inside the expanding boundary).
// Equivalently, these pixels have nonzero SEED IDs.
__global__ void k_init_jumpflood (int2* out, int* ids, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;

	if (my_x >= width || my_y >= height)
		return;

	int idx = my_y*pitch + my_x;
	int2 outval = make_int2(-width, -height);
	
	if (ids[idx] > 0) outval = make_int2(my_x, my_y);

	out[idx] = outval;
}

// Perform one pass of the jump flood algorithm. Updates indirection map for this value of stride.
// Optionally computes a distance transform essentially for free, if dist_out != NULL.
// texref_shirt_2d : current state of the indirection map
__global__ void k_jumpflood_pass (int2* out, int stride, int width, int height, int pitch)
{
	bool valid[8];
	int2 point0;
	int2 point1;
	int2 point2;
	int2 point3;
	int2 point4;
	int2 point5;
	int2 point6;
	int2 point7;
	int2 point8;
	
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;

	if (my_x >= width || my_y >= height)
		return;

	int min_dist2 = INT_MAX;
	int2 min_point;

	bool xmin = my_x-stride >= 0;
	bool xmax = my_x+stride < width;
	bool ymin = my_y-stride >= 0;
	bool ymax = my_y+stride < height;

	valid[0] = xmin && ymin;
	valid[1] = ymin;
	valid[2] = xmax && ymin;
	valid[3] = xmin;
	valid[4] = xmax;
	valid[5] = xmin && ymax;
	valid[6] = ymax;
	valid[7] = xmax && ymax;

	if (valid[0]) point0 = tex2D(texref_int2_2d, my_x-stride, my_y-stride);
	if (valid[1]) point1 = tex2D(texref_int2_2d, my_x, my_y-stride);
	if (valid[2]) point2 = tex2D(texref_int2_2d, my_x+stride, my_y-stride);
	if (valid[3]) point3 = tex2D(texref_int2_2d, my_x-stride, my_y);
	if (valid[4]) point4 = tex2D(texref_int2_2d, my_x+stride, my_y);
	if (valid[5]) point5 = tex2D(texref_int2_2d, my_x-stride, my_y+stride);
	if (valid[6]) point6 = tex2D(texref_int2_2d, my_x, my_y+stride);
	if (valid[7]) point7 = tex2D(texref_int2_2d, my_x+stride, my_y+stride);
	point8 = tex2D(texref_int2_2d, my_x, my_y);
	
	
	{
		int dx = point0.x - my_x;
		int dy = point0.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[0] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point0;
		}
	}
	
	{
		int dx = point1.x - my_x;
		int dy = point1.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[1] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point1;
		}
	}
	
	{
		int dx = point2.x - my_x;
		int dy = point2.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[2] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point2;
		}
	}
	
	{
		int dx = point3.x - my_x;
		int dy = point3.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[3] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point3;
		}
	}
	
	{
		int dx = point4.x - my_x;
		int dy = point4.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[4] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point4;
		}
	}
	
	{
		int dx = point5.x - my_x;
		int dy = point5.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[5] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point5;
		}
	}
	
	{
		int dx = point6.x - my_x;
		int dy = point6.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[6] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point6;
		}
	}
	
	{
		int dx = point7.x - my_x;
		int dy = point7.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (valid[7] && dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point7;
		}
	}
	
	{
		int dx = point8.x - my_x;
		int dy = point8.y - my_y;
		int dist2 = dx*dx + dy*dy;
		if (dist2 < min_dist2)
		{
			min_dist2 = dist2;
			min_point = point8;
		}
	}

	
	int idx = my_y*pitch + my_x;
	out[idx] = min_point;
}

// Exracts the zero level set, the boundary between superpixels. It looks for
// discontinuities in the Superpixel ID assignment map.
// texref_int_2d : superpixel ID map
__global__ void k_extract_level_set(float* out, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;

	if (my_x >= width || my_y >= height)
		return;

	int me = tex2D(texref_int_2d, my_x, my_y);
	int up = tex2D(texref_int_2d, my_x, my_y+1);
	int right = tex2D(texref_int_2d, my_x+1, my_y);

	int idx = my_y*pitch + my_x;
	float outval = 0.0f;

	if ( me != up || me != right)
	{
		outval = 1.0f;
	}

	out[idx] = outval;
}

// Fills any unassigned holes in the image not yet claimed by a superpixel.
// It does this by using the indirection map to calculate a generalized Voronoi
// diagram (each pixel is assigned to its nearest superpixel).
// texref_int_2d : superpixel ID map
// texref_int2_2d : indirection map
__global__ void k_fill_holes(int* out, int width, int height, int pitch)
{
	int my_x = threadIdx.x + blockIdx.x*blockDim.x;
	int my_y = threadIdx.y + blockIdx.y*blockDim.y;

	if (my_x >= width || my_y >= height)
		return;

	int2 nearest_point = tex2D(texref_int2_2d, my_x, my_y);
	int nearest_id = tex2D(texref_int_2d, nearest_point.x, nearest_point.y);
	out[my_y*pitch+my_x] = nearest_id;
}


//
// Host code
//

int get_next_pow2(int n)
{
	return 1 << (int)ceilf(log2f((float)n));
}

// Initialize GPU Device (equivalent to CUT_DEVICE_INIT)
void InitDevice(int dev = 0, bool verb = false) {
    
  int deviceCount;                                                         
  CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                
  if (deviceCount == 0) {                                                  
      fprintf(stderr, "cutil error: no devices supporting CUDA.\n");       
      exit(EXIT_FAILURE);                                                  
  }                                                                        

  if (dev < 0) dev = 0;                                                    
  if (dev > deviceCount-1) dev = deviceCount - 1;                          
  
  cudaDeviceProp deviceProp;                                               
  CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));       
  if (deviceProp.major < 1) {                                              
      fprintf(stderr, "cutil error: device does not support CUDA.\n");     
      exit(EXIT_FAILURE);                                                  
  }                                                                        
  if (verb)    
    fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);

  CUDA_SAFE_CALL(cudaSetDevice(dev));                                      
}

/*
// Writes a floating-point array in device memory to a BMP file as a grayscale image.
void write_float_bmp (const char* file, float* d_ptr)
{
	float* h_out = new float [img_pixels];
	unsigned long* tmp = new unsigned long [img_realpixels];

	cudaMemcpy(h_out, d_ptr, img_pixels*sizeof(float), cudaMemcpyDeviceToHost);

	unsigned char* p = (unsigned char*)tmp;
	for (int y = 0; y < img_height; y++)
	{
		for (int x = 0; x < img_width; x++)
		{
			float val = h_out[y*img_pitch+x];
			val=std::max(0.0f, std::min(1.0f, val));
			*(p++) = (unsigned char)(val * 255.0f);
			*(p++) = (unsigned char)(val * 255.0f);
			*(p++) = (unsigned char)(val * 255.0f);
			*p++ = 255;
		}
	}
	bmp_24_write(file, img_width, img_height, tmp);

	delete[] tmp;
	delete[] h_out;
}

// Writes an array in device memory of 16-bit integers to a BMP file.
// Each possible integer value is mapped to one of 8 colors, with 0 uniquely mapped to black.
void write_int_bmp (const char* file, int* d_ptr)
{
	static unsigned long colors[] = {0xFF8000, 0xFF0000, 0x00FF00, 0x0000FF, 0x00FFFF, 0xFF00FF, 0xFFFF00, 0xFFFFFF};


	int* h_out = new int [img_pixels];
	unsigned long* tmp = new unsigned long [img_realpixels];

	cudaMemcpy(h_out, d_ptr, img_pixels*sizeof(int), cudaMemcpyDeviceToHost);

	unsigned long* p = tmp;
	for (int y = 0; y < img_height; y++)
	{
		for (int x = 0; x < img_width; x++)
		{
			int i = y*img_pitch+x;
			*p++ = (h_out[i] == 0) ? 0x0 : colors[h_out[i] & 0x7];
		}
	}
	bmp_24_write(file, img_width, img_height, tmp);

	delete[] tmp;
	delete[] h_out;
}

// Writes an array in device memory of 16-bit integers to a BMP file.
// Each possible integer value is mapped to one of 8 colors, with 0 uniquely mapped to black.
void write_int_bmp2 (const char* file, int* d_ptr)
{

	int* h_out = new int [img_pixels];
	unsigned long* tmp = new unsigned long [img_realpixels];
  uint16_t out16;

	cudaMemcpy(h_out, d_ptr, img_pixels*sizeof(int), cudaMemcpyDeviceToHost);

	unsigned long* p = tmp;
	for (int y = 0; y < img_height; y++)
	{
		for (int x = 0; x < img_width; x++)
		{
			int i = y*img_pitch+x;
      out16 = 5 * h_out[i] & 0xFFFF;
			*p++ = (h_out[i] == 0) ? 0x0 : out16 + out16<<16;
		}
	}
	bmp_24_write(file, img_width, img_height, tmp);

	delete[] tmp;
	delete[] h_out;
} */

// Copies an image in linear device memory to a Cuda Array texture.
template <class T>
void copy_lin_to_tex(cudaArray* dest, T* src)
{
	CUDA_SAFE_CALL(cudaMemcpy2DToArray(dest, 0, 0, src, img_pitch*sizeof(T), img_width*sizeof(T), img_height, cudaMemcpyDeviceToDevice));
}


void calc_speed_based_on_gradient()
{
	// Get gradient magnitude
	copy_lin_to_tex(d_float_array, d_float_temp1);
	k_gradient_mag<<<STD_CONFIG>>> (d_float_temp1, img_width, img_height, img_pitch);
	CUT_CHECK_ERROR ("kernel failed");

	// Filter the gradient magnitude
	float sigma = floorf(sqrtf((float)img_width * (float)img_height / (float)N_SUPERPIXELS) / 2.5f);
	gaussianFilter(d_float_temp1, d_speed, d_float_temp2, img_pitch, img_height, sigma, 0);
	
	// Calculate speed
	k_calc_speed_based_on_gradient<<<STD_CONFIG>>> (d_float_temp1, d_speed, 1.0f / (sigma*2.5f), d_speed, img_width, img_height, img_pitch);
	
	// Calculate derivatives of speed
	copy_lin_to_tex(d_float_array, d_speed);
	k_gradient_xy<<<STD_CONFIG>>> (d_speed_dx, d_speed_dy, img_width, img_height, img_pitch);

	//if (DEBUG_IMG) write_float_bmp("speed.bmp", d_speed);
}

void place_seeds_and_init_psi()
{
	// Do initial seed placement
	float size_grid_r = 1.0f / sqrtf((float)img_pixels / (float)N_SUPERPIXELS);
	float rows_f = img_height * size_grid_r;
	float cols_f = img_width * size_grid_r;
	int x_dist = (int)ceilf(1.0f/size_grid_r);
	int y_dist = (int)ceilf(1.0f/size_grid_r);
	int rows = (int)ceilf(rows_f - 1);
	int cols = (int)ceilf(cols_f - 1);
	unsigned int max_shift=(unsigned int)ceilf( min((float)x_dist,(float)y_dist)*0.25f - 1.0f );

	// our seed location array is only of size N_SUPERPIXELS. actual number of seeds should be less, in theory.
	assert(rows*cols <= N_SUPERPIXELS);

	// Place the seeds
	k_place_seeds<<< dim3((cols+15)>>4, (rows+15)>>4), dim3(16, 16) >>> (d_seed_locations, x_dist, y_dist, max_shift, rows, cols);
	CUT_CHECK_ERROR ("place seeds kernel failed");

	// Initialize psi (distance func)
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_seed_coords_const, d_seed_locations, rows*cols*sizeof(float2), 0, cudaMemcpyDeviceToDevice));
	k_init_psi<<<STD_CONFIG>>> (d_float_temp1, d_assignment_map, rows*cols, img_width, img_height, img_pitch);	
	CUT_CHECK_ERROR ("init psi kernel failed");

	//if (DEBUG_IMG) write_float_bmp("psi_start.bmp", d_float_temp1);
}


void feature_distance_transform()
{
	int n_passes = (int)ceilf(log2f((float)max((int)img_width, (int)img_height))) - 1;

	// Initialize the indirection map
	copy_lin_to_tex(d_float_array, d_float_temp1);
	k_init_jumpflood<<<STD_CONFIG>>> (d_indirection_map, d_assignment_map, img_width, img_height, img_pitch);

	// Perform logn+1 passes, halving the stride each time
	for (int i = n_passes; i >= 0; i--)
	{
		copy_lin_to_tex(d_int2_array, d_indirection_map);
		k_jumpflood_pass<<<STD_CONFIG>>> (d_indirection_map, 1 << i, img_width, img_height, img_pitch);
	}
	
	/*
	// Another logn passes at stride=1 cleans up any remaining errors
	for (int i = 0; i < n_passes; i++)
	{
		copy_lin_to_tex(d_int2_array, d_indirection_map);
		k_jumpflood_pass<<<STD_CONFIG>>> (d_indirection_map, 1, img_width, img_height, img_pitch);
	}*/

	copy_lin_to_tex(d_int2_array, d_indirection_map);
}

void _initialize(int dev, long width, long height)
{
	// Prep the device
  InitDevice(dev);
  
  img_width = width;
  img_height = height;
  img_realpixels = img_width*img_height;
	img_pitch = (img_width + 31) & (~31);
	img_pixels = img_pitch*img_height;
	img_pixels_pow2 = get_next_pow2(img_pixels);
  
  // Channel formats for textures
	cudaChannelFormatDesc d_float_channel_desc = cudaCreateChannelDesc<float>();
	texref_float_2d.addressMode[0] = cudaAddressModeClamp; 
  texref_float_2d.addressMode[1] = cudaAddressModeClamp; 
  texref_float_2d.filterMode = cudaFilterModePoint; 
  texref_float_2d.normalized = false;

	cudaChannelFormatDesc d_int2_channel_desc = cudaCreateChannelDesc<int2>();
	texref_int2_2d.addressMode[0] = cudaAddressModeClamp; 
  texref_int2_2d.addressMode[1] = cudaAddressModeClamp; 
  texref_int2_2d.filterMode = cudaFilterModePoint; 
  texref_int2_2d.normalized = false;
    
  cudaChannelFormatDesc d_int_channel_desc = cudaCreateChannelDesc<int>();
	texref_int_2d.addressMode[0] = cudaAddressModeClamp; 
  texref_int_2d.addressMode[1] = cudaAddressModeClamp; 
  texref_int_2d.filterMode = cudaFilterModePoint; 
  texref_int_2d.normalized = false;

	// Allocate buffers
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_uchar4_temp1, img_realpixels*sizeof(uchar4)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_uchar1_temp1, img_realpixels*sizeof(uchar1)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_float_temp1, img_pixels*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_float_temp2, img_pixels*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_int_temp1, img_pixels_pow2*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_indirection_map, img_pixels*sizeof(int2)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_assignment_map, img_pixels_pow2*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_speed, img_pixels*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_speed_dx, img_pixels*sizeof(float)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_speed_dy, img_pixels*sizeof(float)));
	CUDA_SAFE_CALL(cudaMallocArray(&d_float_array, &d_float_channel_desc, img_width, img_height));
	CUDA_SAFE_CALL(cudaMallocArray(&d_int2_array, &d_int2_channel_desc, img_width, img_height));
	CUDA_SAFE_CALL(cudaMallocArray(&d_int_array, &d_int_channel_desc, img_width, img_height));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_seed_locations, N_SUPERPIXELS*sizeof(float2)));

	// Bind arrays to texture references
	CUDA_SAFE_CALL(cudaBindTextureToArray(texref_float_2d, d_float_array, d_float_channel_desc));
	CUDA_SAFE_CALL(cudaBindTextureToArray(texref_int2_2d, d_int2_array, d_int2_channel_desc));
	CUDA_SAFE_CALL(cudaBindTextureToArray(texref_int_2d, d_int_array, d_int_channel_desc));

}

// Image in RGB encoded in an unsigned long
void prepare_image(uint64_t* input_img)
{
  // Zero out these two arrays. Their sizes are padded to next-pow2 to simplify the reduction
	// kernel, but those extra padded entries MUST NOT have garbage in them.
	CUDA_SAFE_CALL(cudaMemset(d_assignment_map, 0, img_pixels_pow2*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_int_temp1, 0, img_pixels_pow2*sizeof(int)));

  // Convert image to greyscale
	CUDA_SAFE_CALL(cudaMemcpy(d_uchar4_temp1, input_img, img_realpixels*sizeof(uchar4), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTexture(0, texref_uchar4_1d, d_uchar4_temp1, img_realpixels*sizeof(uchar4)));
	k_rgb2grayf<<<STD_CONFIG>>> (d_float_temp1, img_width, img_height, img_pitch);
	CUT_CHECK_ERROR ("kernel failed");

	// Smooth the image
	for (int i = 0; i < 10; i++)
	{
		copy_lin_to_tex(d_float_array, d_float_temp1);
		k_smooth_image<<<STD_CONFIG>>> (d_float_temp1, 0.1f, img_width, img_height, img_pitch);
	}

	//if (DEBUG_IMG) write_float_bmp("smoothed.bmp", d_float_temp1);
}

// Prepare image if input is already float and normalized
void prepare_image(float *input_img)
{
	// Zero out these two arrays. Their sizes are padded to next-pow2 to simplify the reduction
	// kernel, but those extra padded entries MUST NOT have garbage in them.
	CUDA_SAFE_CALL(cudaMemset(d_assignment_map, 0, img_pixels_pow2*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_int_temp1, 0, img_pixels_pow2*sizeof(int)));
 
  // Copy data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_float_temp1, input_img, img_pixels*sizeof(float), cudaMemcpyHostToDevice));

	// Smooth the image
	for (int i = 0; i < 10; i++)
	{
		copy_lin_to_tex(d_float_array, d_float_temp1);
		k_smooth_image<<<STD_CONFIG>>> (d_float_temp1, 0.1f, img_width, img_height, img_pitch);
	}

	//if (DEBUG_IMG) write_float_bmp("smoothed.bmp", d_float_temp1);
}

// Prepare image if input is uchars 
void prepare_image(unsigned char* input_img)
{
	// Zero out these two arrays. Their sizes are padded to next-pow2 to simplify the reduction
	// kernel, but those extra padded entries MUST NOT have garbage in them.
	CUDA_SAFE_CALL(cudaMemset(d_assignment_map, 0, img_pixels_pow2*sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(d_int_temp1, 0, img_pixels_pow2*sizeof(int)));
 
  // Convert image to internal format 
	CUDA_SAFE_CALL(cudaMemcpy(d_uchar1_temp1, input_img, img_realpixels*sizeof(uchar1), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaBindTexture(0, texref_uchar_1d, d_uchar1_temp1, img_realpixels*sizeof(uchar1)));
	k_gray2grayf<<<STD_CONFIG>>> (d_float_temp1, img_width, img_height, img_pitch);
	CUT_CHECK_ERROR ("kernel failed");

	// Smooth the image
	for (int i = 0; i < 10; i++)
	{
		copy_lin_to_tex(d_float_array, d_float_temp1);
		k_smooth_image<<<STD_CONFIG>>> (d_float_temp1, 0.1f, img_width, img_height, img_pitch);

	}

	//if (DEBUG_IMG) write_float_bmp("smoothed.bmp", d_float_temp1);
}


// Assumption: Someone must have already called "_initialize"
// If input is a float, we assume somebody else has already converted the
// image to grayscale and normalized
template <typename T>
int extractSuperpixels(unsigned int* output, T* input_img)
{

  /*
  unsigned int timer = 0;
  cutCreateTimer(&timer);
	cutStartTimer(timer);
  */

  int iter;
	prepare_image(input_img);
	calc_speed_based_on_gradient();	
	place_seeds_and_init_psi();
	
	// Evolve psi until little forward progress is made
	int old_covered_pixels = do_reduction_count(d_assignment_map, d_int_temp1, img_pixels_pow2);
	for (iter = 0; iter < MAX_ITERATIONS; iter++)
	{
		feature_distance_transform();
	
		copy_lin_to_tex(d_int_array, d_assignment_map);
		k_evolve_psi<<<STD_CONFIG>>> (d_float_temp1, 
                                  d_assignment_map, 
                                  d_speed, 
                                  d_speed_dx, 
                                  d_speed_dy, 
                                  0.5f, 
                                  img_width, 
                                  img_height,
                                  img_pitch);
		
		int new_covered_pixels = do_reduction_count(d_assignment_map, d_int_temp1, img_pixels_pow2);
		float relative_inc = (float)(new_covered_pixels - old_covered_pixels) / (float)img_realpixels;
		old_covered_pixels = new_covered_pixels;
		
    if (relative_inc < 1e-4 && new_covered_pixels >= img_realpixels/2)
			break;

	}
	// Assign any remaining unassigned areas and generate the final superpixel boundaries
	k_fill_holes<<<STD_CONFIG>>> (d_assignment_map, img_width, img_height, img_pitch);
	cudaThreadSynchronize();
  
  /* 
  cutStopTimer(timer);
	printf("time: %f ms\n", cutGetTimerValue(timer));
  */ 
  // Copy data from device memory to output
  cudaMemcpy(output, d_assignment_map, img_pixels*sizeof(int), cudaMemcpyDeviceToHost);
	return iter;
}

void _cleanup()
{
	cudaFree(d_uchar4_temp1);
	cudaFree(d_uchar1_temp1);
	cudaFree(d_float_temp1);
	cudaFree(d_float_temp2);
	cudaFree(d_int_temp1);
	cudaFree(d_indirection_map);
	cudaFree(d_assignment_map);
	cudaFree(d_speed);
	cudaFree(d_speed_dx);
	cudaFree(d_speed_dy);
	cudaFree(d_seed_locations);
	cudaFreeArray(d_float_array);
	cudaFreeArray(d_int2_array);
	cudaFreeArray(d_int_array);

}

// Template explicit instantiation
template int extractSuperpixels(unsigned int* output, unsigned char* input_img);
template int extractSuperpixels(unsigned int* output, float* input_img);
template int extractSuperpixels(unsigned int* output, uint64_t* input_img);


