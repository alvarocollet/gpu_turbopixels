///////////////////////////////////////////////////////////////////////////////
//
//  Turbopixels source code.
//
//  Copyright (c) 2011 Carnegie Mellon University
//  Author: Alvaro Collet (alvaro.collet@gmail.com)
//
///////////////////////////////////////////////////////////////////////////////

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


