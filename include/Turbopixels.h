///////////////////////////////////////////////////////////////////////////////
//
//  Turbopixels header file.
//
//  Copyright (c) 2011 Carnegie Mellon University
//  Author: Alvaro Collet (alvaro.collet@gmail.com)
//
///////////////////////////////////////////////////////////////////////////////
#ifndef __TURBOPIXELS_H__
#define __TURBOPIXELS_H__

#include "turbopix.h"
#include <stdint.h>
#include <stdio.h>

namespace tpix {

  /**
   * \brief Maximum number of superpixels we search for
   */
  static const int MAX_SUPERPIXELS = N_SUPERPIXELS;

  class Turbopixels {

  private:
    int dev;
    int width;
    int height;

  public:

    /**
     * \brief Default constructor
     */
    Turbopixels ();

    /**
     * \brief Constructor with custom image size
     * \param width image width, in pixels
     * \param height image height, in pixels
     *
     */
    Turbopixels (int width, int height);

    /**
     * \brief Full constructor: custom image size, custom GPU
     * \param dev gpu device to use. use 0 if you don't care...
     * \param img_width image width, in pixels
     * \param img_height image height, in pixels
     *
     */
    Turbopixels (int dev, int width, int height);

    /**
     * \brief destructor.
     */
    ~Turbopixels ();

    /**
     * \brief Reshape image size. It deallocates and reallocates all memory
     * \param width image width, in pixels
     * \param height image height, in pixels
     * \output none
     *
     */
    void
    reshape (int width, int height);

    /**
     * \brief Reshape image size. It deallocates and reallocates all memory
     * \param dev gpu device to use. use 0 if you don't care...
     * \param img_width image width, in pixels
     * \param img_height image height, in pixels
     * \output none
     *
     */
    void
    reshape (int dev, int width, int height);

    /**
     * \brief deallocate all used memory and general cleanup
     *
     */
    void
    cleanup ();

    /**
     * \brief Process image to create superpixels
     * \param src image is a vector of grayscale values, normalized [0,1]
     * \param dst (output) vector of superpixel IDs, one for every pixel
     * \output number of iterations needed for convergence
     *
     */
    int
    process (unsigned int* dst, float *src);

    /**
     * \brief Process image to create superpixels
     * \param src image is a vector of grayscale values in range [0, 255]
     * \param dst (output) vector of superpixel IDs, one for every pixel
     * \output number of iterations needed for convergence
     *
     */
    int
    process (unsigned int* dst, unsigned char* src);

    /**
     * \brief Process image to create superpixels
     * \param src image is a vector of color values packed in a 64 bit int,
     *        s.t. src[i] = {0,0,0,0,0,b,g,r}
     * \param dst (output) vector of superpixel IDs, one for every pixel
     * \output number of iterations needed for convergence
     *
     */
    int
    process (unsigned int* dst, uint64_t *src);

  };
}
#endif
