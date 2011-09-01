/*
 * Turbopixels.h
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

#ifndef __TURBOPIXELS_H__
#define __TURBOPIXELS_H__

#include "turbopix.h"
#include <stdint.h>
#include <stdio.h>

namespace tpix {

  /**
   * \brief Maximum number of superpixels we can ever search for
   */
  static const int MAX_SUPERPIXELS = _MAX_SUPERPIXELS;

  class Turbopixels {

  private:
    int dev;
    int width;
    int height;
    unsigned int nSuperpixels;

  public:

    /**
     * \brief Default constructor
     */
    Turbopixels ();

    /**
     * \brief Constructor with custom image size
     * \param width image width, in pixels
     * \param height image height, in pixels
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     *
     */
    Turbopixels (int width, int height, unsigned int nSuperpixels);

    /**
     * \brief Full constructor: custom image size, custom GPU
     * \param dev gpu device to use. use 0 if you don't care...
     * \param img_width image width, in pixels
     * \param img_height image height, in pixels
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     *
     */
    Turbopixels (int dev, int width, int height, unsigned int nSuperpixels);

    /**
     * \brief destructor.
     */
    ~Turbopixels ();

    /**
     * \brief Reshape image. It deallocates and reallocates all memory
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     * \output none
     *
     */
    void
    reshape (unsigned int nSuperpixels);

    /**
     * \brief Reshape image size. It deallocates and reallocates all memory
     * \param width image width, in pixels
     * \param height image height, in pixels
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     * \output none
     *
     */
    void
    reshape (int width, int height, unsigned int nSuperpixels);

    /**
     * \brief Reshape image size. It deallocates and reallocates all memory
     * \param dev gpu device to use. use 0 if you don't care...
     * \param img_width image width, in pixels
     * \param img_height image height, in pixels
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     * \output none
     *
     */
    void
    reshape (int dev, int width, int height, unsigned int nSuperpixels);

    /**
     * \brief deallocate all used memory and general cleanup
     *
     */
    void
    cleanup ();

    /**
     * \brief Process image to create superpixels
     * \param[in] src image is a vector of grayscale values, normalized [0,1]
     * \param[out] dst vector of superpixel IDs, one for every pixel
     * \output number of iterations needed for convergence
     *
     */
    int
    process (unsigned int* dst, float *src);

    /**
     * \brief Process image to create superpixels
     * \param[in] src image is a vector of grayscale values in range [0, 255]
     * \param[out] dst vector of superpixel IDs, one for every pixel
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     * \output number of iterations needed for convergence
     *
     */
    int
    process (unsigned int* dst, unsigned char* src);

    /**
     * \brief Process image to create superpixels
     * \param[in] src image is a vector of color values packed in a 64 bit int,
     *        s.t. src[i] = {0,0,0,0,0,b,g,r}
     * \param[out] dst vector of superpixel IDs, one for every pixel
     * \param nSuperpixels Desired number of superpixels to compute (actual
     *   number might vary). Default: 1000.
     * \output number of iterations needed for convergence
     *
     */
    int
    process (unsigned int* dst, uint64_t *src);

  };
}
#endif
