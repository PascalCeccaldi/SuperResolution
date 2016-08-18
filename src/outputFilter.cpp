#include <highgui.h>
#include "outputFilter.hh"

OutputFilter::OutputFilter() : filter(true)
{}


void* OutputFilter::operator()(void* tok)
{
  std::pair<IplImage*, IplImage*>* pair
      = static_cast<std::pair<IplImage*, IplImage*>*> (tok);

  IplImage* img = pair->first;

  cvShowImage("StreamFilter", img);

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
