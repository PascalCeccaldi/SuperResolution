#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <vector>


#define ASSERT_EX(condition, statement) \
    do { \
        if (!(condition)) { statement; assert(condition); } \
    } while (false)


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

  ASSERT_EX((is >= 0 && is < src->rows), std::cout << "IS" << is << std::endl);
  ASSERT_EX((js >= 0 && is < src->cols), std::cout << "JS" << js << std::endl);
  ASSERT_EX((id >= 0 && id + 3 < dst->rows), std::cout << "ID" << id + 3 << std::endl);
  ASSERT_EX((jd >= 0 && jd + 3 < dst->cols), std::cout << "JD" << jd + 3 << std::endl);

  Vec3b color = src->at<Vec3b>(Point(is, js));
  for (int c = 0; c < src->channels(); ++c)
  {
    dst->at<uchar>(Point(id, jd + c)) = color[0];
  }
}

void setCell(Mat* dst, int row, int col, Vec3b value)
{
  for (int c = 0; c < 3; ++c)
  {
    dst->at<uchar>(row, col + c) = value[c];
  }
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
      if (isInBounds(src, row + i, col + j)) {
        copyCell(src, dst, row + i, col + j, sample_index, index);
      }
      else
      {
        Vec3b color(0, 0, 0);
        setCell(dst, sample_index, index, color);
      }
      index += 3;
    }
  }
}

Mat buildSampleData(std::vector<Mat>* pyrH, std::vector<Mat>* pyrL)
{

  Mat first = *pyrH->begin();
  Mat samples(getSampleSize(pyrH), 28, CV_8UC1);


  int channels = first.channels();

  int sample_index = 0;

  for(int l = 0; l < pyrH->size(); ++l)
  {
    Mat* hi = &pyrH->at(l);
    Mat* li = &pyrL->at(l + 1);

    std::cout << " DIM HI " << hi->cols << "  " << hi->rows << " " << hi->channels() << std::endl;
    std::cout << " DIM LI " << li->cols << "  " << li->rows << " " << li->channels() << std::endl;
    for(int  i = 0; i < hi->rows - 1; ++i)
    {
      for (int  j = 0; j < hi->cols - 1; ++j)
      {
        copyCell(hi, &samples, i, j, sample_index, 0);
        setNeighborhood(li, &samples, i, j, sample_index);
        sample_index++;
      }
    }
  }

  return samples;
}

std::vector<Mat> getUx(Mat means)
{
  std::vector<Mat> ux;
  for (int i = 0; i < means.rows; i++)
  {
    Mat uix(means, Rect(0, i, 25, 1));
    ux.push_back(uix);
  }
  return ux;
}

std::vector<Mat> getUy(Mat means)
{
  std::vector<Mat> uy;
  for (int i = 0; i < means.rows; i++)
  {
    Mat uiy(means, Rect(25, i, 3, 1));
    uy.push_back(uiy);
  }
  return uy;
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



  EM model(7, EM::COV_MAT_GENERIC, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 10));

  std::cout << "model created" << std::endl;
  model.train(samples);

  if (model.isTrained()){
    std::cout << "model trained" << std::endl;
  } else {
    std::cout << "model not trained" << std::endl;
  }

  const vector<Mat>& covs  = model.get<vector<Mat>>("covs");
  const Mat means  = model.get<Mat>("means");
  const Mat weights  = model.get<Mat>("weights");
  std::vector<Mat> ux = getUx(means);
  std::vector<Mat> uy = getUy(means);


  std::vector<Mat> sxx;
  std::vector<Mat> sxy;
  std::vector<Mat> syx;
  std::vector<Mat> syy;


  for (unsigned int i = 0; i < covs.size(); i++)
  {
    Mat sixx(covs.at(i), Rect(0, 0, 25, 25));
    sxx.push_back(sixx);
    Mat sixy(covs.at(i), Rect(25, 0, 3, 25));
    sxy.push_back(sixy);
    Mat siyx(covs.at(i), Rect(0, 25, 25, 3));
    syx.push_back(siyx);
    Mat siyy(covs.at(i), Rect(25, 25, 3, 3));
    syy.push_back(siyy);
  }


  std::cout << covs.at(0).rows << " " << covs.at(0).cols << std::endl;
  std::cout << means.rows << " " << means.cols << std::endl;
  std::cout << weights.rows << " " << weights.cols << std::endl;
  std::cout << weights << std::endl;


  return 0;

}
