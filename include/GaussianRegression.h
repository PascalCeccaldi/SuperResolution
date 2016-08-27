//
// Created by Paul KLEIN on 25/08/2016.
//
#pragma once

#include "includes.h"

using namespace cv;

class GaussianRegressor {

  public:

    GaussianRegressor(EM model);
    Vec3b estimate(Mat sample);


  private:

    std::vector<Mat> buildMeansX(Mat means);
    std::vector<Mat> buildMeansY(Mat means);
    std::vector<double> computeBetas(Mat sample);
    Vec3b computeMuysx(Mat sample, Mat meanX, Mat meanY, Mat covYX, Mat covXX);
    double computeProbPdf(Mat samples, Mat cov, Mat mean);


    unsigned int n_component;
    std::vector<Mat> meansX;
    std::vector<Mat> meansY;
    std::vector<Mat> covsXX;
    std::vector<Mat> covsXY;
    std::vector<Mat> covsYX;
    std::vector<Mat> covsYY;
    std::vector<double> weights;
};
