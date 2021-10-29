# Halide_Panorama
Panorama using Halide

# Requirement
Halide: 12.0.0 or above

armadillo: 

# Methods
1.Apply Harris corner method to find corners in images. 

2.Find correspondent corners of each adjacent images. 

3.Apply Ransanc method to find best Homography matrix of images from correspondent corners.

4.Transform images with Homography matrix.

5.Combine all transformed images.

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
