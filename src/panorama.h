#pragma once

#include <Halide.h>
#include <stdio.h>

using namespace Halide;

Func harris_corners(Func im, Expr width, Expr height, int max_diam, float k, double sigma,
                    double factor, double boundary_size);

Func compute_feature(Func im, Expr width, Expr height, Func corners, 
                     float sigma_blur_descriptor, int radius_descriptor);

Func find_correspondences(Func feature_1, Func feature_2, Func corner_1, Func corner_2, int max_diam,
                    int num_tile_x, int num_tile_y, int radius_descriptor, float threshold);

Func ransac(Func correspondence, int num_tile_x, int num_tile_y, int Niter, float epsilon);

Func stitch(Func im1, Func im2, int w1, int h1, int w2, int h2, Func H, Func bbox);

Func visualize_harris(Func im, Func corner, int num_tile_x, int num_tile_y, int width, int height);

Func visualize_correspondence(Func im, Func correspondence, int num_tile_x, int num_tile_y, 
                                int n, int width ,int height);