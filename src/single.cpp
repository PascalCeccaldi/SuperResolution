#include "SRSingleImageGMM.hh"


int main(void) {

  float scale_factor = 2;
  int levels = 3;
  int n_component = 3;

  Mat h0;
  h0 = imread("013.jpg", CV_LOAD_IMAGE_COLOR);

  if(! h0.data )
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  Mat Hr = SRSingleImageGMM::predict(h0, scale_factor, levels, n_component);

  imwrite("SrGMM.jpg", Hr);

  return 0;

}
