#pragma once
#include "includes.h"

using namespace cv;

class SRSingleImageGMM {

  public:

    static Mat predict(Mat h0, float scale_factor, int levels, int n_component);

  private:

    static Mat buildSampleData(std::vector<Mat>* pyrH, std::vector<Mat>* pyrL);
    static Mat getNeighborhood(Mat* src, int row, int col);
    static void setNeighborhood(Mat* src, Mat* dst, int row, int col, int sample_index);
    static void setCell(Mat* dst, int row, int col, Vec3b value);
    static void copyCell(Mat* src, Mat* dst, int is, int js, int id, int jd);
    static bool isInBounds(Mat* src, int i, int j);
    static int getSampleSize(std::vector<Mat>* pyrH);
    static std::vector<Mat>* buildLPyramid(std::vector<Mat>* pyrH, float scale_factor);
    static std::vector<Mat>* buildHPyramid(Mat h0, float scale_factor, int levels);

};