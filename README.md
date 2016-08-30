# Video Super-resolution

The goal of this project is to implement a quality-enhancing tool
of a video stream using machine learning.

## Reference

It implements an algorithm proposed here [Single Image Super-Resolution - He et al.](http://f4k.dieei.unict.it/proceedings/ICPR2012/media/files/1411.pdf)

## Result

### Single image output :

![alt est01](https://github.com/PascalCeccaldi/ParallelImages/blob/master/img/est01.jpg)

### Bicubic interpolated :

![alt interp01](https://github.com/PascalCeccaldi/ParallelImages/blob/master/img/interp01.jpg)


### Zoomed

![alt zoom_est01](https://github.com/PascalCeccaldi/ParallelImages/blob/master/img/zoom_est01.jpg)
>>>>>>> Updated Readme

![alt zoom_interp01](https://github.com/PascalCeccaldi/ParallelImages/blob/master/img/zoom_interp01.jpg)


## Notes

* The output is not satisfying, the image saturates where the original pixels are close to black or white; causing null or infinite probabilities for one component of the GMM in the estimation phase. Some quality has been gained but it is roughly equivalent to aplying unsharp masking.
* The current implementation does not benefit from the nature of the content, i.e. it does not use the continuity of the frames. Moreover, the performance is very poor since each pixel has to be predicted independently. We could also use patches to learn the conversion function from the interpolated image but introducing new degrees of freedom to the model would probably lead to under-fitting the data.
* Some other papers proposing algorithms offer more visually satisfying results but that seem less natural and do not show SSIM or PNSR scores - [Super-Resolution from a Single Image - Glasner et al.](http://www.wisdom.weizmann.ac.il/~vision/single_image_SR/files/single_image_SR.pdf)

 
