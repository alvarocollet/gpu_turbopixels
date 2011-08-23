/*
 * GPU_TURBOPIXELS
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

GPU_Turbopixels implements the superpixel computation from:

Levinshtein, A., Stere, A., Kutulakos, K. N., Fleet, D. J., Dickinson, S. J., &
Siddiqi, K. (2009). "TurboPixels: fast superpixels using geometric flows." IEEE
Transactions on Pattern Analysis and Machine Intelligence, 31(12), 2290-7.

GPU_Turbopixels provides a very fast computation of superpixels with good
spatial localization, which means that the generated superpixels are often
round-ish and do not have the elongated/spindly shapes common with other
approaches such as the Felzenszwalb-Huttenlocher segmentation.

I (Alvaro) did not create the core cuda files of this implementation. I found
them as an anonymous publish, and I spent the time to marginally optimize the
code, reorganized with proper header files, wrote a C++ wrapper class and a
simple example file to use std::vectors as input and output. The example also
shows how Turbopixels interacts with OpenCV 2.  My best bet for who the
original author of the CUDA files is is Alex Radionov, who published a technical
report on computing Turbopixels on the GPU. However, I have not been able to
confirm this and the code did not have any copyright/license notice. If you are
the author of the CUDA files for turbopixels, please let me know and I will add
you as the author.
 
If you use Turbopixels, you should cite Levinshtein paper. Also, if you use
*this* implementation (GPU_Turbopixels), you should cite the following paper
(for which I created this code):

Collet, A., Srinivasa, S. S., & Hebert, M. (2011). "Structure Discovery in
Multi-modal Data: a Region-based Approach." IEEE International Conference on
Robotics and Automation. 


INSTALLATION AND DEPENDENCIES
-----------------------------
This code requires CUDA 3.0 or higher (CUDA 4.0 is recommended). You can freely download CUDA from NVIDIA's webpage.

I have tested this software with Ubuntu 10.04 and two different Nvidia GPUs, a
GTX 260 and a Ti 550. Please let me know if you succeed/run into any issues
with other configurations.

For convenience, I wrap this code as a ROS package, which makes linking to other packages extremely straightforward. However, the code does *NOT* depend on ROS.The same can be said about OpenCV: I provide an example (src/example.cpp) which loads an image, converts it to an std::vector which the Turbopixels c++ wrapper understands, and then saves it to disk. However, there are no OpenCV dependencies in the Turbopixels class.

I attach my own cmake file to find CUDA and its dependencies (FindCudaComps.cmake). If the default version cannot find the CUDA SDK, you can help CMAKE by creating the following environment variable (example given for bash):
export NVSDKCUDA_ROOT=PATH_TO_CUDA_SDK

To link this library from other code, check the lflags and cppflags from
manifest.xml (if you use ROS, just add the package dependency to your
manifest.xml).  In case you want a static library, be warned that I have had a
lot of trouble linking to a static library that uses CUDA (always end up with
unresolved external symbols). You will save yourself some pain if you link your
code against the dynamic library libturbopixels.so.

Please let me know if you find any issues with this code.

Alvaro Collet
alvaro.collet@gmail.com
