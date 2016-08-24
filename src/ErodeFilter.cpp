#include "ErodeFilter.hh"

ErodeFilter::ErodeFilter() : filter (false)
{}

void* ErodeFilter::operator()(void* elt)
{
  std::pair<IplImage*, IplImage*>* pair
    = static_cast<std::pair<IplImage*, IplImage*>*> (elt);
  IplImage* img = pair->first;


  IplImage* out;
  out = cvCreateImage(cvSize(img->width * 4, img->height * 4), img->depth, img->nChannels);

  cvResize(img, out, CV_INTER_CUBIC);

  //out = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, img->nChannels);


  //cvErode(img, out, 0, 2);

  std::pair<IplImage*, IplImage*>* pair2
    = new std::pair<IplImage*, IplImage*>(out, pair->second);
  return pair2;
}

std::string ErodeFilter::get_name ()
{
  return "Erode filter";
}
