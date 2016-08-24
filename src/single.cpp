#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;

std::vector<Mat>* buildHPyramid(Mat h0, float scale_factor, int levels)
{
  std::vector<Mat>* pyrH = new std::vector<Mat>();
  pyrH->push_back(h0);
  Mat temp = h0;
  for (int i = 1; i <= levels + 2; ++i) {
    Mat hmi;
    int h = (int) (temp.cols / scale_factor);
    int w = (int) (temp.rows / scale_factor);
    pyrDown(temp, hmi, Size(h, w));
    pyrH->push_back(hmi);
    temp = hmi;
  }

  return pyrH;
}


std::vector<Mat>* buildLPyramid(std::vector<Mat>* pyrH, float scale_factor)
{

  std::vector<Mat>* pyrL = new std::vector<Mat>();
  int i = 0;
  for (Mat hmi: *pyrH)
  {
    Mat lmi;
    int h = (int) (hmi.cols * scale_factor);
    int w = (int) (hmi.rows * scale_factor);

    resize(hmi, lmi, Size(h, w), CV_INTER_CUBIC);
    pyrL->push_back(lmi);
  }
  pyrH->pop_back();

  return pyrL;
}


int main(int argc, char** argv) {

  std::string window_name = "image";
  float scale_factor = 2;
  int levels = 2;

  Mat h0;
  h0 = imread("012.jpg", CV_LOAD_IMAGE_COLOR);

  if(! h0.data )
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  std::vector<Mat>* pyrH = buildHPyramid(h0, scale_factor, levels);
  std::vector<Mat>* pyrL = buildLPyramid(pyrH, scale_factor);

  std::cout << pyrH->size() << " " << pyrL->size() << std::endl;

  /*
  namedWindow( window_name, WINDOW_AUTOSIZE );
  imshow( window_name, hm2 );
  waitKey(0);
  */

  return 0;

}