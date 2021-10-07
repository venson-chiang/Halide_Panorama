#pragma once

#include <Halide.h>
#include <stdio.h>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

using namespace Halide;

Func calculate_luminance(Func input, std::string name);

Func calculate_gaussian(float sigma, int dim);

Func blur_gaussian(Func input, float sigma, std::string name);

Func calculate_tensor(Func im, Expr width, Expr height, double sigma, double factor_sigma);

Func generate_4p(Expr n);

Func merge_sort(Func input, int tile_x, int tile_y);

Func num_valid_corners(Func sort, int max_len);

Func compute_transformed_bbox(int width, int height, Func H);

Func compute_bbox(Func box, Func inv_H);

Func bbox_union(std::vector<Func> cb_box);

Func apply_homography(Func im1, Func im2, Func H, int w2, int h2, Func bbox);

Func apply_finalH(Func im, Func H, int w, int h);

Func color_correct(Func im1, Func im2, Expr w1, Expr h1, Expr w2, Expr h2, Func weight1, Func weight2);

Func prodH(Func H1, Func H2);

Func prodP(Func H, Expr x, Expr y);

Func onesH();

Func translate_correspondence(Func correspondence, Func inv_H);

Func inverse_homography(Func H);

Func downscale(Func im, int n); 

Func calculate_weight(int w, int h);

Func calculate_highfrequency(Func im, Func low);

Func stitch_high(Func high1, Func high2, Func transweight1, Func transweight2);

Func stitch_low(Func low1, Func low2, Func transweight1, Func transweight2);

Func to_uint8(Func im);

Func combine_weight(Func weight1, Func weight2);

Func color_match(Func im1, Func im2, Func weight1, Func weight2, int width, int height, Func inv_H1, Func inv_H2, int wmax, int hmax, Func box);

Func color_correction(Func im1, Func im2, Func weight1, Func weight2, int max_w, int max_h);