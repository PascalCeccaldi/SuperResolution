#include "GaussianRegression.hh"
#include "SRSingleImageGMM.hh"

using namespace cv;


std::vector<Mat>* SRSingleImageGMM::buildHPyramid(Mat h0, float scale_factor, int levels)
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


std::vector<Mat>* SRSingleImageGMM::buildLPyramid(std::vector<Mat>* pyrH, float scale_factor)
{

  std::vector<Mat>* pyrL = new std::vector<Mat>();
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


int SRSingleImageGMM::getSampleSize(std::vector<Mat>* pyrH)
{
  int size = 0;
  for (Mat hmi: *pyrH)
  {
    size += hmi.rows * hmi.cols;
  }
  return size;
}


void SRSingleImageGMM::copyCell(Mat* src, Mat* dst, int is, int js, int id, int jd)
{
  Vec3b color = src->at<Vec3b>(is, js);
  for (int c = 0; c < src->channels(); ++c)
  {
    dst->at<double>(id, jd + c) = color[c];
  }
}


void SRSingleImageGMM::setCell(Mat* dst, int row, int col, Vec3b value)
{
  for (int c = 0; c < 3; ++c)
  {
    dst->at<double>(row, col + c) = value[c];
  }
}


bool SRSingleImageGMM::isInBounds(Mat* src, int i, int j)
{
  if (i > 0 && j > 0 && i < src->rows && j < src->cols)
    return true;
  return false;
}


void SRSingleImageGMM::setNeighborhood(Mat* src, Mat* dst, int row, int col, int sample_index)
{
  int index = 0;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j) {
      if (i == 1 && j == 1)
        continue;
      if (isInBounds(src, row + i, col + j)) {
        copyCell(src, dst, row + i, col + j, sample_index, index + 3);
      }
      else
      {
        Vec3b color(0, 0, 0);
        setCell(dst, sample_index, index + 3, color);
      }
      index += 3;
    }
  }
}


Mat SRSingleImageGMM::getNeighborhood(Mat* src, int row, int col)
{
  int index = 0;
  Mat res(24, 1, CV_64FC1);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j) {
      if (i == 1 && j == 1)
        continue;
      if (isInBounds(src, row + i, col + j)) {
        Vec3b color = src->at<Vec3b>(row + i, col + j);
        res.at<double>(index) = color[0];
        res.at<double>(index + 1) = color[1];
        res.at<double>(index + 2) = color[2];
      }
      else
      {
        res.at<double>(index) = 0;
        res.at<double>(index + 1) = 0;
        res.at<double>(index + 2) = 0;
      }
      index += 3;
    }
  }
  return res;
}


Mat SRSingleImageGMM::buildSampleData(std::vector<Mat>* pyrH, std::vector<Mat>* pyrL)
{

  Mat first = *pyrH->begin();
  Mat samples(getSampleSize(pyrH), 27, CV_64FC1);


  int sample_index = 0;

  for(unsigned int l = 0; l < pyrH->size(); ++l)
  {
    Mat* hi = &pyrH->at(l);
    Mat* li = &pyrL->at(l + 1);

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


Mat SRSingleImageGMM::predict(Mat h0, float scale_factor, int levels, int n_component)
{
  std::vector<Mat>* pyrH = buildHPyramid(h0, scale_factor, levels);
  std::vector<Mat>* pyrL = buildLPyramid(pyrH, scale_factor);

  Mat samples = buildSampleData(pyrH, pyrL);

  EM model(n_component, EM::COV_MAT_GENERIC, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1));

  Mat log_likelihoods;
  model.train(samples, log_likelihoods);

  GaussianRegressor* gr = new GaussianRegressor(model);

  Mat Lm = *pyrL->begin();
  Mat Hr(Lm.rows, Lm.cols, CV_8UC3, double(0));
  for (int i = 0; i < Lm.rows; i++)
  {
    for (int j = 0; j < Lm.cols; j++)
    {
      Mat sample = getNeighborhood(&Lm, i, j);
      Vec3d px = gr->estimate(sample);

      if (px[0] > 255)
        px[0] = 255.0;
      if (px[1] > 255)
        px[1] = 255.0;
      if (px[1] > 255)
        px[1] = 255.0;
      if (px[2] > 255)
        px[2] = 255.0;
      if (px[0] < 0)
        px[0] = 0;
      if (px[1] < 0)
        px[1] = 0;
      if (px[1] < 0)
        px[1] = 0;
      if (px[2] < 0)
        px[2] = 0;

      Vec3b v0((uchar) round(px[0]), (uchar) round(px[1]), (uchar) round(px[2]));
      Hr.at<Vec3b>(i, j) = v0;
    }
    imshow("HR Result", Hr);
    waitKey(1);
  }

  return Hr;
}
