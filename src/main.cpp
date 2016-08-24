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
  //capture = cvCreateCameraCapture(0);
  cvNamedWindow("img", CV_WINDOW_FREERATIO | CV_WINDOW_NORMAL );

  capture = cvCreateFileCapture("test.mp4");

  if (!cvGrabFrame(capture))
  {              // capture a frame
    printf("Could not grab a frame from capture1\n\7");
    exit(0);
  }
  //capture = cvCreateCameraCapture(0);


  int height = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
  int width = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
  double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
  CvVideoWriter* writer = cvCreateVideoWriter("out.avi", CV_FOURCC('I', 'Y', 'U', 'V'),
                                              fps, cvSize(width, height), true);

  InputFilter input (capture);
  pipeline.add_filter(input);
  //pipeline.add_filter(*new ErodeFilter());
  OutputFilter output(writer);
  pipeline.add_filter(output);
  pipeline.run(threads);
  pipeline.clear();

  cvReleaseCapture(&capture);
  cvReleaseVideoWriter(&writer);
  return 0;
}
