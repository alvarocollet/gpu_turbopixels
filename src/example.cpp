/*
 * example.cpp
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

/*
 * EXAMPLE OF USAGE OF CLASS TURBOPIXELS
 */
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <vector>

int main(int argc, char** argv){

  tpix::Turbopixels tpix;
  unsigned long* h_input_image;
  unsigned long img_width;
  long img_height;
  long img_pitch;

	if (argc < 2)
	{
		printf ("Usage: %s <input image>\n", argv[0]);
		return 0;
	}

  cv::Mat img;
  img = cv::imread(argv[1], 0);
  cv::Size sz = img.size();
  img_width = sz.width;
  img_height = sz.height;

  unsigned long img_size = img_width*img_height;
  std::vector<unsigned int> out(img_size);
  std::vector<unsigned char> input_imgc(img_size);

  // cv::MatIterator_<unsigned char> it = img.begin<unsigned char>();
  // cv::MatIterator_<unsigned char> it_end = img.end<unsigned char>();
  input_imgc.assign( img.begin<unsigned char>(), img.end<unsigned char>() );

  // Timing information
  struct timespec tStep, tEnd;
  clock_gettime(CLOCK_REALTIME, &tStep);

  // Execute turbopixels
  int nIter = tpix.process( (unsigned int*) &out[0], (unsigned char*) &input_imgc[0] );

  // Display how much time we spent
  clock_gettime(CLOCK_REALTIME, &tEnd);
  float tstep = ( (tEnd.tv_sec -  tStep.tv_sec)*1000000000LL + tEnd.tv_nsec -  tStep.tv_nsec )/1000000.;
  printf("Time spent: %f ms, in %d iterations.\n", tstep, nIter);

  // Shuffle the superpixels indexes, so that we can see something in output
  unsigned int maxVal = (unsigned int) *(std::max_element( out.begin(), out.end() ));
  unsigned int minVal = (unsigned int) *(std::min_element( out.begin(), out.end() ));
  std::vector<int> shuff( maxVal ); 
  std::vector<int>::iterator it = shuff.begin();
  for (int i = 0; it < shuff.end(); i++, ++it)
    *it = i;
  std::random_shuffle( shuff.begin(), shuff.end() );
  cout << "Number of superpixels: " << maxVal << endl;
  
  // Copy image to output
  cv::MatIterator_<unsigned char> m = img.begin<unsigned char>();
  for (int i = 0; m < img.end<unsigned char>(); ++i, ++m) 
    *m = shuff[ out[i] % maxVal ];

  // Write output to file
  cv::imwrite("output.jpg", img);

  // Show output in a window
  cv::namedWindow("superpixels", CV_WINDOW_AUTOSIZE);
  cv::imshow("superpixels", img);
  cv::waitKey();
  return 0;
}

