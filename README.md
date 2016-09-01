# Video Super-resolution

The goal of this project is to implement a quality-enhancing tool
of a video stream using machine learning.

## Reference

It implements an algorithm proposed here [Single Image Super-Resolution - He et al.](http://f4k.dieei.unict.it/proceedings/ICPR2012/media/files/1411.pdf)


## Notes

* The output is not satisfying, Some quality has been gained but it is roughly equivalent to aplying unsharp masking.
* The current implementation does not benefit from the nature of the content, i.e. it does not use the continuity of the frames. Moreover, the performance is very poor since each pixel has to be predicted independently. We could also use patches to learn the conversion function from the interpolated image but introducing new degrees of freedom to the model would probably lead to under-fitting the data.
* Some other papers proposing algorithms offer more visually satisfying results but that seem less natural and do not show SSIM or PNSR scores - [Super-Resolution from a Single Image - Glasner et al.](http://www.wisdom.weizmann.ac.il/~vision/single_image_SR/files/single_image_SR.pdf)

 
