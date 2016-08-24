#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
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
    pyrL->insert(pyrL->begin(), lmi);
  }
  pyrH->pop_back();

  return pyrL;
}

int getSampleSize(std::vector<Mat>* pyrH)
{
  int size = 0;
  for (Mat hmi: *pyrH)
  {
    size += hmi.rows * hmi.cols;
  }
  return size;
}

void copyCell(Mat* src, Mat* dst, int is, int js, int id, int jd)
{
  Vec3b color = src->at<Vec3b>(Point(is, js));
  dst->at<Vec3b>(Point(id, jd)) = color;
}

void setCell(Mat* dst, int row, int col, Vec3b value)
{
  dst->at<Vec3b>(Point(row, col)) = value;
}

bool isInBounds(Mat* src, int i, int j)
{
  if (i > 0 && j > 0 && i < src->rows && j < src->cols)
    return true;
  return false;
}


int setNeighborhood(Mat* src, Mat* dst, int row, int col, int sample_index)
{
  int index = 0;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j) {
      if (i == 1 && j == 1)
        continue;
      if (isInBounds(src, row + i, col + i)) {
        copyCell(src, dst, sample_index, index, row + i, col + i);
      }
      else
      {
        Vec3b color(0, 0, 0);
        setCell(dst, sample_index, index, color);
      }
      index++;
    }
  }
}

Mat buildSampleData(std::vector<Mat>* pyrH, std::vector<Mat>* pyrL)
{

  Mat first = *pyrH->begin();
  Mat samples(getSampleSize(pyrH), 9, CV_8UC3);


  int channels = first.channels();

  int sample_index = 0;

  for(int l = 0; l < pyrH->size(); ++l)
  {
    Mat hi = pyrH->at(l);
    Mat li = pyrL->at(l + 1);
    imshow("hi", hi);
    waitKey(0);

    std::cout << " DIM " << hi.cols << "  " << hi.rows << " " << hi.channels() << std::endl;
    for(int  i = 0; i < hi.rows; ++i)
    {
      for (int  j = 0; j < hi.cols; ++j)
      {
        copyCell(&hi, &samples, i, j, sample_index, 0);
        setNeighborhood(&li, &samples, i, j, sample_index);
        sample_index++;
      }
    }
  }

  //imshow("samples", samples);
  //waitKey(0);

  return samples;
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

  Mat samples = buildSampleData(pyrH, pyrL);

  EM em = EM(7);
  em.train(samples);
  if (em.isTrained()){
    std::cout << "model trained" << std::endl;
  } else {
    std::cout << "model not trained" << std::endl;
  }
  em.predict(samples);
  /*
  namedWindow( window_name, WINDOW_AUTOSIZE );
  imshow( window_name, hm2 );
  waitKey(0);
  */

  return 0;

}
