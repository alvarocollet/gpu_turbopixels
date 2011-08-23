///////////////////////////////////////////////////////////////////////////////
//
//  Turbopixels header file.
//
//  Copyright (c) 2011 Carnegie Mellon University
//  Author: Alvaro Collet (alvaro.collet@gmail.com)
//
///////////////////////////////////////////////////////////////////////////////
#ifndef __TURBOPIX_H__
#define __TURBOPIX_H__

#include <stdint.h>

// I know, it's a disgrace that N_SUPERPIXELS is not configurable, but whoever
// did the turbopix.h code didn't think about this.
#define N_SUPERPIXELS 1000
#define DEBUG_IMG false
#define MAX_ITERATIONS 500

template <typename T>
int extractSuperpixels(unsigned int* output, T* input_img);

template <typename T>
int extractSuperpixelsWithTiming(unsigned int* output, T* input_img);

// Initialize arrays (should be called in constructor)
void _initialize(int dev, long img_width, long img_height);

// Free arrays (should be called in destructor)
void _cleanup();


#endif


