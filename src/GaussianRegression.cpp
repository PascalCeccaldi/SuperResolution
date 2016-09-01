#include "GaussianRegression.hh"

using namespace cv;

GaussianRegressor::GaussianRegressor(EM model)
{
  vector<Mat> covs  = model.get<vector<Mat>>("covs");
  Mat means  = model.get<Mat>("means");
  Mat wgt(model.get<Mat>("weights"));


  n_component = (unsigned int) wgt.cols;

  for (int i = 0; i < wgt.cols; i++)
  {
    weights.push_back(wgt.at<double>(i));
  }

  meansX = buildMeansX(means);
  meansY = buildMeansY(means);


  for (unsigned int i = 0; i < covs.size(); i++)
  {
    Mat sixx(covs.at(i), Rect(Point(0, 0), Size(24, 24)));
    assert(sixx.rows == sixx.cols);

    covsXX.push_back(sixx);
    Mat sixy(covs.at(i), Rect(Point(24, 0), Size(3, 24)));
    covsXY.push_back(sixy);
    Mat siyx(covs.at(i), Rect(Point(0, 24), Size(24, 3)));
    covsYX.push_back(siyx);
    Mat siyy(covs.at(i), Rect(Point(24, 24), Size(3, 3)));
    assert(siyy.rows == siyy.cols);
    covsYY.push_back(siyy);
  }
}

std::vector<Mat> GaussianRegressor::buildMeansX(Mat means)
{
  std::vector<Mat> ux;
  for (int i = 0; i < means.rows; i++)
  {
    Mat uix(means, Rect(0, i, 24, 1));
    ux.push_back(uix);
  }
  return ux;
}

std::vector<Mat> GaussianRegressor::buildMeansY(Mat means)
{
  std::vector<Mat> uy;
  for (int i = 0; i < means.rows; i++)
  {
    Mat uiy(means, Rect(24, i, 3, 1));
    uy.push_back(uiy);
  }
  return uy;
}

Vec3d GaussianRegressor::estimate(Mat sample)
{
  std::vector<double> betas = computeBetas(sample);
  Vec3d estimate(0, 0, 0);

  for (unsigned int i = 0; i < betas.size(); i++)
  {
    Vec3d Muiysx = computeMuysx(sample, meansX.at(i), meansY.at(i), covsYX.at(i), covsXX.at(i));

    estimate[0] += betas.at(i) * Muiysx[0];
    estimate[1] += betas.at(i) * Muiysx[1];
    estimate[2] += betas.at(i) * Muiysx[2];
  }

  return estimate;
}

double GaussianRegressor::computeProbPdf(Mat samples, Mat cov, Mat mean)
{
  Mat pos_cov;
  if (determinant(cov) == 0) {
    Mat offset = Mat::eye(cov.rows, cov.cols, cov.type()) * FLT_EPSILON;
    pos_cov = cov + offset;
  } else {
    pos_cov = cov;
  }
  int dim = pos_cov.rows;
  double det = determinant(pos_cov);
  //std::cout << "DET " << det << std::endl << std::endl;
  double scale = 1.0 / ((pow(2 * M_PI, dim / 2.0) * pow(det, 0.5)));
  //std::cout << "SCALE " << scale << std::endl << std::endl;
  Mat invcov = pos_cov.inv();
  Mat tmp1 = samples - mean.t();
  Mat tmp2 = tmp1.t() * invcov * tmp1;
  return exp(log(scale) + (- 1 / 2) * tmp2.at<double>(0, 0));
}

Vec3d GaussianRegressor::computeMuysx(Mat sample, Mat meanX, Mat meanY, Mat covYX, Mat covXX)
{
  Mat tmp1 = sample - meanX.t();
  Mat tmp2 = covYX * covXX.inv();
  Mat tmp3 = tmp2 * tmp1;
  Mat Muysx = meanY + tmp3.t();

  Vec3d ret(Muysx.at<double>(0), Muysx.at<double>(1), Muysx.at<double>(2));

  return ret;
}

std::vector<double> GaussianRegressor::computeBetas(Mat sample)
{
  double denom = 0;
  std::vector<double> betas;

  for (unsigned int i = 0; i < n_component; i++)
  {
    denom += weights[i] * computeProbPdf(sample, covsXX.at(i), meansX.at(i));
  }

  for (unsigned int i = 0; i < n_component; i++)
  {
    double beta = weights[i] * computeProbPdf(sample, covsXX.at(i), meansX.at(i));

    betas.push_back(beta / denom);
  }

  return betas;
}

