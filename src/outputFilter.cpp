#include <highgui.h>
#include "outputFilter.hh"
#include <stdio.h>
#include <opencv2/opencv.hpp>

OutputFilter::OutputFilter(CvVideoWriter* writer) :
    filter(true),
    writer(writer)
{}


void* OutputFilter::operator()(void* tok)
{
  std::pair<IplImage*, IplImage*>* pair
      = static_cast<std::pair<IplImage*, IplImage*>*> (tok);

  IplImage* img = pair->first;

  cvShowImage("img", img);

  char buf[100];
  sprintf(buf, "frames/%03d.jpg", counter); // Make sure directory 'frames' exists.
  counter++;
  // cvWrite

  cvSaveImage(buf, img);

  //cvWriteFrame(writer, img);

  if (cvWaitKey(1) != -1)
  {
    cvDestroyAllWindows();
    exit(0);
  }

  return 0;
}

std::string OutputFilter::get_name()
{
    return "Output Filter";
}
