#include "GaussianRegression.hh"
#include "SRSingleImageGMM.hh"
#include "tbb/tbb.h"

using namespace cv;


std::vector<Mat>* SRSingleImageGMM::buildHPyramid(Mat h0, float scale_factor, int levels, int isPara)
{
  std::vector<Mat>* pyrH = new std::vector<Mat>();
  pyrH->push_back(h0);
  Mat temp = h0;
  if (isPara < 2){
    for (int i = 1; i <= levels + 2; i++) {
      Mat hmi;
      int h = (int) (temp.cols / scale_factor);
      int w = (int) (temp.rows / scale_factor);
      pyrDown(temp, hmi, Size(h, w));
      pyrH->push_back(hmi);
      temp = hmi;
    }
  } else {
    tbb::parallel_for(1, levels + 3, [&](int i){
      Mat hmi;
      int h = (int) (temp.cols / scale_factor);
      int w = (int) (temp.rows / scale_factor);
      pyrDown(temp, hmi, Size(h, w));
      pyrH->push_back(hmi);
      temp = hmi;
    });
  }


  return pyrH;
}


std::vector<Mat>* SRSingleImageGMM::buildLPyramid(std::vector<Mat>* pyrH, float scale_factor, int isPara)
{

  std::vector<Mat>* pyrL = new std::vector<Mat>();
  size_t index = 0;
  if (isPara < 2){
    for (Mat hmi: *pyrH)
    {
      Mat lmi;
      int h = 0;
      int w = 0;
      if (index - 1 < pyrH->size())
      {
        Mat sz = pyrH->at(index - 1);
        h = sz.cols;
        w = sz.rows;
      }
      else
      {
        h = (int) (hmi.cols * scale_factor);
        w = (int) (hmi.rows * scale_factor);
      }
      std::cout << w << " " << h << std::endl;
      resize(hmi, lmi, Size(h, w), CV_INTER_CUBIC);
      pyrL->push_back(lmi);
      index++;
    }
  } else {
    tbb::parallel_for_each(pyrH->begin(), pyrH->end(), [&](Mat hmi){
      Mat lmi;
        int h = 0;
        int w = 0;
        if (index - 1 < pyrH->size())
        {
          Mat sz = pyrH->at(index - 1);
          h = sz.cols;
          w = sz.rows;
        }
        else
        {
          h = (int) (hmi.cols * scale_factor);
          w = (int) (hmi.rows * scale_factor);
        }
      resize(hmi, lmi, Size(h, w), CV_INTER_CUBIC);
      pyrL->push_back(lmi);
        index++;
    });
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
  if (i >= 0 && j >= 0 && i < src->rows && j < src->cols)
    return true;
  return false;
}


void SRSingleImageGMM::setNeighborhood(Mat* src, Mat* dst, int row, int col, int sample_index)
{
  int index = 0;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++) {
      if (i == 1 && j == 1)
        continue;
      if (isInBounds(src, row + i - 1, col + j - 1)) {
        copyCell(src, dst, row + i - 1, col + j - 1, sample_index, index + 3);
      }
      else
      {
        //std::cout << "SI " << row + i - 1 << " " << col + j - 1 << std::endl;
        //std::cout << "DI " << src->rows << " " << src->cols << std::endl;
        //std::cout << " OUT OF BOUNDS " << std::endl;
      }
      index += 3;
    }
  }
}


Mat SRSingleImageGMM::getNeighborhood(Mat* src, int row, int col)
{
  int index = 0;
  Mat res(24, 1, CV_64FC1);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++) {
      if (i == 1 && j == 1)
        continue;
      if (isInBounds(src, row + i - 1, col + j - 1)) {
        Vec3b color = src->at<Vec3b>(row + i - 1, col + j - 1);
        res.at<double>(index) = (double) color[0];
        res.at<double>(index + 1) = (double) color[1];
        res.at<double>(index + 2) = (double) color[2];
      }
      else
      {
        //std::cout << " OUT OF BOUNDS " << std::endl;
        res.at<double>(index) = 0.0;
        res.at<double>(index + 1) = 0.0;
        res.at<double>(index + 2) = 0.0;
      }
      index += 3;
    }
  }
  return res;
}


Mat SRSingleImageGMM::buildSampleData(std::vector<Mat>* pyrH, std::vector<Mat>* pyrL, int isPara)
{

  Mat first = *pyrH->begin();
  Mat samples(getSampleSize(pyrH), 27, CV_64FC1);


  int sample_index = 0;

  if (isPara < 1){
    for(unsigned int l = 0; l < pyrH->size(); ++l)
    {
      Mat* hi = &pyrH->at(l);
      Mat* li = &pyrL->at(l + 1);




      for(int  i = 1; i < hi->rows - 1; i++)
      {
        for (int  j = 1; j < hi->cols - 1; j++)
        {
            copyCell(hi, &samples, i, j, sample_index, 0);
            std::cout << "HI " << hi->rows << " " << hi->cols << std::endl;
            std::cout << "LI " << li->rows << " " << li->cols << std::endl;
            setNeighborhood(li, &samples, i, j, sample_index);
            sample_index++;
        }
      }
    }
  } else {
    tbb::parallel_for( tbb::blocked_range<int>(0, int(pyrH->size())),
    [&]( const tbb::blocked_range<int> r ) {
        for(int l=r.begin(), l_end=r.end(); l<l_end; l++){
          Mat* hi = &pyrH->at(l);
          Mat* li = &pyrL->at(l + 1);
          tbb::parallel_for( tbb::blocked_range2d<int>(0, hi->rows, 0, hi->cols),
          [&]( const tbb::blocked_range2d<int> r ) {
            for(int i=r.rows().begin(), i_end=r.rows().end(); i < i_end; i++){
              for(int j=r.cols().begin(), j_end=r.cols().end(); j < j_end; j++){
                if (i == 0 || i == hi->rows - 1 || j == hi->cols - 1  || j == 0)
                {
                  continue;
                }
                else {
                  copyCell(hi, &samples, i, j, sample_index, 0);
                  setNeighborhood(li, &samples, i, j, sample_index);
                  sample_index++;
                }
              }
            }
          });
        }
      });
  }

  return samples;
}


Mat SRSingleImageGMM::predict(Mat h0, float scale_factor, int levels, int n_component, int isPara)
{
  std::vector<Mat>* pyrH = buildHPyramid(h0, scale_factor, levels, isPara);
  std::vector<Mat>* pyrL = buildLPyramid(pyrH, scale_factor, isPara);

  Mat samples = buildSampleData(pyrH, pyrL, isPara);

  EM model(n_component, EM::COV_MAT_GENERIC, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 100, 1));

  Mat log_likelihoods;
  model.train(samples, log_likelihoods);

  GaussianRegressor* gr = new GaussianRegressor(model);

  Mat Lm = *pyrL->begin();
  Mat Hr(Lm.rows, Lm.cols, CV_8UC3, 255.0);
  if (isPara < 1)
  {
    for (int i = 0; i < Lm.rows; i++)
    {
      for (int j = 0; j < Lm.cols; j++)
      {

        if (i == 0 || i == Lm.rows - 1 || j == Lm.cols - 1  || j == 0)
        {
          Vec3b px = Lm.at<Vec3b>(i, j);
          Hr.at<Vec3b>(i, j) = px;
        }
        else {
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
      }
      imshow("HR Result", Hr);
      waitKey(1);
    }
  } else {

    //std::cout << Lm.rows << " " << Lm.cols << std::endl;
    //std::cout << Hr.rows << " " << Hr.cols << std::endl;

    tbb::parallel_for(tbb::blocked_range2d<int>(0, Lm.rows, 0, Lm.cols),
    [&](const tbb::blocked_range2d<int> r) {
        for(int i = r.rows().begin(), i_end = r.rows().end(); i < i_end; i++){
            for(int j = r.cols().begin(), j_end = r.cols().end(); j < j_end; j++){
              if (i < 2 || i >= Lm.rows - 2 || j >= Lm.cols - 2  || j < 2)
              {
                Vec3b px = Lm.at<Vec3b>(i, j);
                Hr.at<Vec3b>(i, j) = px;
              }
              else {
                Mat sample = SRSingleImageGMM::getNeighborhood(&Lm, i, j);
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
            }
        }
    });
  }
  return Hr;
}
