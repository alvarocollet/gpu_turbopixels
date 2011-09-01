/*
 * turbopix.h
 *
 *  Created on: Aug 20, 2011
 *      Author: Alvaro Collet (acollet@cs.cmu.edu)
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

// -------------------------------------------------------------------------- //
// PRIVATE HEADER FILE
// -------------------------------------------------------------------------- //
#ifndef __TURBOPIX_H__
#define __TURBOPIX_H__

#include <stdint.h>

// The maximum number of superpixels you can ever get is 8192 (64KB of type float2).
#define _MAX_SUPERPIXELS 8000
#define DEBUG_IMG false
#define MAX_ITERATIONS 500

template <typename T>
int extractSuperpixels(unsigned int* output, T* input_img);

template <typename T>
int extractSuperpixelsWithTiming(unsigned int* output, T* input_img);

// Initialize arrays (should be called in constructor)
void _initialize(int dev, long img_width, long img_height, unsigned int nSuperpixels);

// Free arrays (should be called in destructor)
void _cleanup();


#endif


