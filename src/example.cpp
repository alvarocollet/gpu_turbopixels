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

  cv::flip(img, img, 0);
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

  cv::flip(img, img, 0);
  cv::imwrite("output.jpg", img);

  cv::namedWindow("superpixels", CV_WINDOW_AUTOSIZE);
  cv::imshow("superpixels", img);
  cv::waitKey();
  return 0;
}

