/*
 * Turbopixels.cpp
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


#include "Turbopixels.h"

namespace tpix {
  // Default guess: image is 640 by 480. If you want some other
  // size, DO NOT use this constructor!
  Turbopixels::Turbopixels() : dev(0), width(640), height(480){
    _initialize(this->dev, this->width, this->height);
  }

  Turbopixels::Turbopixels(int width, int height) : dev(0), width(width), height(height){
    _initialize(this->dev, width, height);
  }

  Turbopixels::Turbopixels(int dev, int width, int height): dev(dev), width(width), height(height){
    _initialize(dev, width, height);
  }

  Turbopixels::~Turbopixels(){
    _cleanup();
  }

  void Turbopixels::reshape(int width, int height){
    // First clean, then re-initialize structures
    _cleanup();
    _initialize(this->dev, width, height);
  }

  void Turbopixels::reshape(int dev, int width, int height){
    // First clean, then re-initialize structures
    _cleanup();
    _initialize(dev, width, height);
  }

  int Turbopixels::process(unsigned int* dst, uint64_t* src){
    return extractSuperpixels(dst, src);  
  }

  int Turbopixels::process(unsigned int* dst, unsigned char* src){
    return extractSuperpixels(dst, src);  
  }
}


