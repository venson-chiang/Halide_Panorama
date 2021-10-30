# Halide_Panorama
Panorama using Halide

# Requirement
1.Halide: 12.0.0 or above

2.armadillo: 10.6.2

# Methods
1.Compute harris corners, and features around corners. 

2.Find the best pair corners of every two adjacent images. 

3.Apply Ransanc method to find best Homography matrix of images from pair corners.

4.Calculate boundary box of output image.

4.Transform images with Homography matrix to output boundary box coordinate.

5.stitch all transformed images.

# Input images
Input images are reference from https://github.com/aivanov93/panorama/tree/master/images

<img src="https://github.com/venson-chiang/Halide_Panorama/blob/main/images/vancouver-6.png" width="18%" height="18%"> <img src="https://github.com/venson-chiang/Halide_Panorama/blob/main/images/vancouver-5.png" width="18%" height="18%"> <img src="https://github.com/venson-chiang/Halide_Panorama/blob/main/images/vancouver-4.png" width="18%" height="18%"> <img src="https://github.com/venson-chiang/Halide_Panorama/blob/main/images/vancouver-3.png" width="18%" height="18%"> <img src="https://github.com/venson-chiang/Halide_Panorama/blob/main/images/vancouver-2.png" width="18%" height="18%"> 

# Output image
<img src="https://github.com/venson-chiang/Halide_Panorama/blob/main/output_images/vancouver-panorama.png" width="100%" height="100%">

More exposure correct images are in https://github.com/venson-chiang/Halide_Panorama/tree/main/output_images

# Usage
1. Change HALIDE_DISTRIB_PATH to yours in Makefile.inc
```
HALIDE_DISTRIB_PATH ?= /mnt/d/Software/Halide-12/distrib 
```
2. Run Makefile 
```
make test
```

# Reference
Panorama model is reference to https://github.com/aivanov93/panorama
