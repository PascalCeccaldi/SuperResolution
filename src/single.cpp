#include "GaussianRegression.h"

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
  Vec3b color = src->at<Vec3b>(is, js);
  for (int c = 0; c < src->channels(); ++c)
  {
    dst->at<uchar>(id, jd + c) = color[c];
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


void setNeighborhood(Mat* src, Mat* dst, int row, int col, int sample_index)
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

Mat getNeighborhood(Mat* src, int row, int col)
{
  int index = 0;
  Mat res(24, 1, CV_64FC1);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j) {
      if (i == 1 && j == 1)
        continue;
      if (isInBounds(src, row + i, col + j)) {
        res.at<uchar>(index) = src->at<uchar>(row + i, col + j);
      }
      else
      {
        res.at<uchar>(index) = 0;
      }
      index += 3;
    }
  }
  return res;
}

Mat buildSampleData(std::vector<Mat>* pyrH, std::vector<Mat>* pyrL)
{

  Mat first = *pyrH->begin();
  Mat samples(getSampleSize(pyrH), 27, CV_8UC1);


  int sample_index = 0;

  for(unsigned int l = 0; l < pyrH->size(); ++l)
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







int main(void) {

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

  int n_component = 5;
  EM model(n_component, EM::COV_MAT_GENERIC, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 25, 100));


  Mat log_likelihoods;
  std::cout << "model created" << std::endl;
  model.train(samples, log_likelihoods);

  if (model.isTrained()){
    std::cout << "model trained" << std::endl;
  } else {
    std::cout << "model not trained" << std::endl;
  }

  for(int j = 0; j < n_component; j++ )
  {
    std::cout << log_likelihoods.at<double>(j) << std::endl;
  }

  GaussianRegressor* gr = new GaussianRegressor(model);

  Mat Lm = *pyrL->begin();
  Mat Hr(Lm.rows, Lm.cols, CV_8UC3);
  for (int i = 0; i < Lm.rows; i++)
  {
    for (int j = 0; j < Lm.cols; j++)
    {
      Mat sample = getNeighborhood(&Lm, i, j);
      Vec3b px = gr->estimate(sample);
      Hr.at<Vec3b>(i, j) = px;
    }
  }


  // Some testing to see what is the effect on the output image

  bool flattening = true;
  bool clipping = false;

  if (flattening) {
    Mat bgr[3];   //destination array
    split(Hr, bgr);

    std::vector<Mat> colors;

    for (int i = 0; i < 3; i++) {
      double min, max;
      minMaxLoc(bgr[i], &min, &max);
      double OldRange = (max - min);
      double NewRange = 255.0;
      bgr[i] = (((bgr[i] - min) * NewRange) / OldRange);
      colors.push_back(bgr[i]);
    }
    merge(colors, Hr);
  }

  if (clipping) {
    for (int i = 0; i < Hr.rows; i++) {
      for (int j = 0; j < Hr.cols; j++) {
        Vec3b color = Hr.at<Vec3b>(i, j);
        if (color[0] < 0)
          color[0] = 0;
        if (color[1] < 0)
          color[1] = 0;
        if (color[2] < 0)
          color[2] = 0;
        if (color[0] > 255)
          color[0] = 255;
        if (color[1] > 255)
          color[1] = 0;
        if (color[2] > 255)
          color[2] = 255;
        Hr.at<Vec3b>(i, j) = color;
      }
    }
  }


  imshow("Interp", Lm);
  imshow("HR Result", Hr);
  waitKey(0);

  return 0;

}
