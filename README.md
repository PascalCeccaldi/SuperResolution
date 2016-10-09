# Single Image Super-resolution


The goal of this project is to implement a quality-enhancing tool
of a image using machine learning.

We use a Gaussian Mixture Model to learn a conversion fonction between a spline interpolated image and the desired image via EM.
The training data is the concatenation of the rgb components of a pixel and it's 8-pixel neighborhood of an interpolated image of the same size. Training data is built from different layers of a gaussian pyramid and an bicubic-interpolation pyramid.

We then use the learned joint distribution to predict output pixels from it's neighborhood in a scaled interpolated image.

## Reference

It implements a model of prediction proposed here [Single Image Super-Resolution - He et al.](http://f4k.dieei.unict.it/proceedings/ICPR2012/media/files/1411.pdf)


## Notes

* The output is not satisfying, Some quality has been gained but it is roughly equivalent to aplying unsharp masking.
* The initialization of the gmm does not permit the EM to make it converge to an appropriate representqtion of the joint distribution.
    --> We obtain worse results than spline interpolation at PSNR and SSIM metrics
 
