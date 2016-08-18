#include "inputFilter.hh"

InputFilter::InputFilter(CvCapture* cap)
    : filter(true),
      capture (cap)
{}


void* InputFilter::operator()(void*)
{
  if (cvGrabFrame(capture))
  {
    IplImage* img = cvRetrieveFrame (capture);
    std::pair<IplImage*, IplImage*>* pair = new std::pair<IplImage*, IplImage*>(img, 0);
    return pair;
  }
  else
    return 0;
}

std::string InputFilter::get_name()
{
    return "Input Filter";
}
