#include "cv.h"

#include "inputFilter.hh"
#include "outputFilter.hh"
#include "ErodeFilter.hh"

#include "highgui.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/pipeline.h"

using namespace cv;



int main(int argc, char** argv)
{
  size_t threads = 3;

  tbb::pipeline pipeline;

  CvCapture* capture;
  capture = cvCreateCameraCapture(0);

  InputFilter input (capture);
  pipeline.add_filter(input);
  pipeline.add_filter(*new ErodeFilter());
  OutputFilter output;
  pipeline.add_filter(output);
  pipeline.run(threads);
  pipeline.clear();

  cvReleaseCapture(&capture);
  return 0;
}
