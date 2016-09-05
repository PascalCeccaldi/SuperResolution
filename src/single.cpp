#include "SRSingleImageGMM.hh"
#include <sys/time.h>

double getPSNR ( const Mat& I1, const Mat& I2);
Scalar getMSSIM( const Mat& I1, const Mat& I2);

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

// first arg: 0 no parallel, other parallel
int main(int argc, char** argv) {

  float scale_factor = 2;
  int levels = 3;
  int n_component = 3;

  Mat h0, h1;
  h1 = imread("013.jpg", CV_LOAD_IMAGE_COLOR);

  if(! h1.data )
  {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }

  pyrDown(h1, h0, Size(h1.cols / scale_factor, h1.rows / scale_factor));

  int isPara = atoi(argv[1]);
  timestamp_t t0 = get_timestamp();
  Mat Hr = SRSingleImageGMM::predict(h0, scale_factor, levels, n_component, isPara);
  timestamp_t t1 = get_timestamp();
  double secs = (t1 - t0) / 1000000.0L;
  imwrite("SrGMM.jpg", Hr);

  std::cout << "time for prediction : " << secs << std::endl;

  std::cout << "PSNR our Solution = " << getPSNR(h1, Hr) << std::endl;
  std::cout << "SSIM our Solution = " << getMSSIM(h1, Hr) << std::endl;

  Mat bic;
  resize(h0, bic, Size(h1.cols, h1.rows), CV_INTER_CUBIC);
  std::cout << "PSNR bicubic = " << getPSNR(h1, bic) << std::endl;
  std::cout << "SSIM bicubic = " << getMSSIM(h1, bic) << std::endl;

  return 0;

}

double getPSNR(const Mat& I1, const Mat& I2)
{
 Mat s1;
 absdiff(I1, I2, s1);       // |I1 - I2|
 s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
 s1 = s1.mul(s1);           // |I1 - I2|^2

 Scalar s = sum(s1);         // sum elements per channel

 double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

 if( sse <= 1e-10) // for small values return zero
     return 0;
 else
 {
     double  mse =sse /(double)(I1.channels() * I1.total());
     double psnr = 10.0*log10((255*255)/mse);
     return psnr;
 }
}

Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
 const double C1 = 6.5025, C2 = 58.5225;
 /***************************** INITS **********************************/
 int d     = CV_32F;

 Mat I1, I2;
 i1.convertTo(I1, d);           // cannot calculate on one byte large values
 i2.convertTo(I2, d);

 Mat I2_2   = I2.mul(I2);        // I2^2
 Mat I1_2   = I1.mul(I1);        // I1^2
 Mat I1_I2  = I1.mul(I2);        // I1 * I2

 /***********************PRELIMINARY COMPUTING ******************************/

 Mat mu1, mu2;   //
 GaussianBlur(I1, mu1, Size(11, 11), 1.5);
 GaussianBlur(I2, mu2, Size(11, 11), 1.5);

 Mat mu1_2   =   mu1.mul(mu1);
 Mat mu2_2   =   mu2.mul(mu2);
 Mat mu1_mu2 =   mu1.mul(mu2);

 Mat sigma1_2, sigma2_2, sigma12;

 GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
 sigma1_2 -= mu1_2;

 GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
 sigma2_2 -= mu2_2;

 GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
 sigma12 -= mu1_mu2;

 ///////////////////////////////// FORMULA ////////////////////////////////
 Mat t1, t2, t3;

 t1 = 2 * mu1_mu2 + C1;
 t2 = 2 * sigma12 + C2;
 t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

 t1 = mu1_2 + mu2_2 + C1;
 t2 = sigma1_2 + sigma2_2 + C2;
 t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

 Mat ssim_map;
 divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

 Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
 return mssim;
}
