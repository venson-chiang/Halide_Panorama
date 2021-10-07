#include "utils.h"
#include <Halide.h>
#define ARMA_WARN_LEVEL 1
#include <armadillo>

using namespace Halide;
using namespace Halide::ConciseCasts;

// calculate luminance of image
Func calculate_luminance(Func input, std::string name) {

    Func output(name);
    Var x, y;
    output(x, y) = input(x, y, 0)*0.3f + input(x, y, 1)*0.6f + input(x, y, 2)*0.1f;

    // Schedule
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

// calculate_gaussian kernel -- Return gaussian kernel with the given sigma
Func calculate_gaussian(float sigma, int dim) {

    Func output("output_calculate_gaussian"), Val;
    RDom r(0, 2*dim+1, 0, 2*dim+1);
    Var x, y;

    Val(x, y) = exp(-(pow(x - dim, 2) + pow(y - dim, 2)) / (2.f * sigma * sigma));
    Expr gauss_sum = sum(Val(r.x, r.y));
    output(x, y) = Val(x, y) / gauss_sum;

    // Schedule
    Val.compute_root();
    output.compute_root();

    return output;
}

// Appies gaussian blur to the given Halide, Default sigma=1
Func blur_gaussian(Func input, float sigma, std::string name) {
    
    Func output(name);

    int dim = (int)floor(sigma * 3);
    Func gaussian = calculate_gaussian(sigma, dim);

    RDom r(0, 2*dim+1, 0, 2*dim+1);
    Var x, y, c;

    if (input.dimensions() == 3) {
        output(x, y, c) = sum(gaussian(r.x, r.y) * input(x+r.x-dim, y+r.y-dim, c));
    } else {
        output(x, y) = sum(gaussian(r.x, r.y) * input(x+r.x-dim, y+r.y-dim));
    }

    // Schedule
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

// Calculate gradient Tensor
Func calculate_tensor(Func im, Expr width, Expr height, double sigma, double factor_sigma) {
    
    //Func luminance;
    Func gradient_x("gradient_x");
    Func gradient_y("gradient_y");
    Func tensor("tensor");
    Func output("output_calculate_tensor");

    Var x, y, c;

    // calculate gradient
    gradient_x(x, y) = im(x-1, y-1) * (-1.0f/12) + im(x+1, y-1) * (1.0f/12) +
                       im(x-1, y  ) * (-2.0f/12) + im(x+1, y  ) * (2.0f/12) +
                       im(x-1, y+1) * (-1.0f/12) + im(x+1, y+1) * (1.0f/12);

    gradient_y(x, y) = im(x-1, y-1) * (-1.0f/12) + im(x-1, y+1) * (1.0f/12) +
                       im(x  , y-1) * (-2.0f/12) + im(x  , y+1) * (2.0f/12) +
                       im(x+1, y-1) * (-1.0f/12) + im(x+1, y+1) * (1.0f/12); 

    // calculate tensor
    tensor(x, y, c) = select(c == 0, gradient_x(x, y) * gradient_x(x, y),
                             c == 1, gradient_x(x, y) * gradient_y(x, y),
                                     gradient_y(x, y) * gradient_y(x, y));
    // blur tensor
    output = blur_gaussian(tensor, sigma * factor_sigma, "tensor_blurred");

    // Schedule
    gradient_x.compute_root().parallel(y).vectorize(x, 16);
    gradient_y.compute_root().parallel(y).vectorize(x, 16);
    tensor.compute_root().parallel(y).vectorize(x, 16);

    return output; 
}

// generate 4 random points
Func generate_4p(Expr n) {

    Func a0, a1, a2, a3;
    Func b0, b1, b2, b3;
    Func c0, c1, c2, c3;
    Func d0, d1, d2, d3;
    Func output("output_4_random_points");

    Var x("x"), c("c");

    a0(x) = random_uint() % (n-3);
    a1(x) = random_uint() % (n-3);
    a2(x) = random_uint() % (n-3);
    a3(x) = random_uint() % (n-3);

    b0(x) = min(a0(x), a1(x));
    b1(x) = max(a0(x), a1(x));
    b2(x) = min(a2(x), a3(x));
    b3(x) = max(a2(x), a3(x));

    c0(x) = min(b0(x), b2(x));
    c1(x) = max(b0(x), b2(x));
    c2(x) = min(b1(x), b3(x));
    c3(x) = max(b1(x), b3(x));

    d0(x) = c0(x);
    d1(x) = min(c1(x), c2(x));
    d2(x) = max(c1(x), c2(x));
    d3(x) = c3(x);

    output(x, c) = select(c==0, d0(x), c==1, d1(x)+1, c==2, d2(x)+2, d3(x)+3);

    output.bound(c, 0, 4).compute_root().reorder(c, x).parallel(x).vectorize(c);

    return output;
}

// calculate sorted input 2D matrix to seperate valid corners(>=0) and unvalid corners(=-1)
// Sorting method reference https://github.com/halide/Halide/blob/master/test/performance/sort.cpp
Func merge_sort(Func input, int tile_x, int tile_y) {
    
    Func flat_input("flat_input");
    Func result("sort_result");

    Var x, y, n;
    flat_input(n) = select(n < tile_x * tile_y - 1,
                            select(input(n % tile_x, n / tile_x, 0) != -1, n, -1),
                            -1);
    flat_input.compute_root();

    // First gather the input into 2D array of width four where each row is sorted
    {
    Expr a0 = flat_input(4 * y);
    Expr a1 = flat_input(4 * y + 1);
    Expr a2 = flat_input(4 * y + 2);
    Expr a3 = flat_input(4 * y + 3);

    Expr b0 = max(a0, a1);
    Expr b1 = min(a0, a1);
    Expr b2 = max(a2, a3);
    Expr b3 = min(a2, a3);

    a0 = max(b0, b2);
    a1 = min(b0, b2);
    a2 = max(b1, b3);
    a3 = min(b1, b3);

    b0 = a0;
    b1 = max(a1, a2);
    b2 = min(a1, a2);
    b3 = a3;

    result(x, y) = select(x==0, b0, x==1, b1, x==2, b2, b3);
    result.compute_root();
    }
    // build up to the total size, merging each pair of rows
    for (int chunck_size = 4; chunck_size < tile_x*tile_y; chunck_size *= 2) {
        
        Func merge_rows("merge_rows");
        RDom r(0, chunck_size * 2);

        merge_rows(x, y) = Tuple(0, 0, cast(flat_input.value().type(), 0));

        Expr candidate_a = merge_rows(r-1, y)[0];
        Expr candidate_b = merge_rows(r-1, y)[1];
        Expr valid_a = candidate_a < chunck_size;
        Expr valid_b = candidate_b < chunck_size;
        Expr value_a = result(clamp(candidate_a, 0, chunck_size-1), 2*y  );
        Expr value_b = result(clamp(candidate_b, 0, chunck_size-1), 2*y+1);

        merge_rows(r, y) = tuple_select(valid_a && (!valid_b || (value_a > value_b)),
                                        Tuple(candidate_a+1, candidate_b  , value_a),
                                        Tuple(candidate_a  , candidate_b+1, value_b));
        merge_rows.compute_root();
        
        result = lambda(x, y, merge_rows(x, y)[2]);
    }

    return lambda(x, result(x, 0));
}

// calculate number of valid corners
Func num_valid_corners(Func sort, int max_len) {

    Var x;
    Func output("length_of_valid_corners");
    RDom r(0, max_len);
    r.where(sort(r.x) != -1);
    output(x) = sum((r.x+1)/(r.x+1));

    output.compute_root().bound(x, 0, 1);

    return output;
}

// calculate homography from the given list of corresponding points
// homography method refernece to https://www.itread01.com/content/1547026589.html
// arma solve function reference to http://arma.sourceforge.net/docs.html#solve
extern "C" DLLEXPORT int calculate_homography(halide_buffer_t * input, halide_buffer_t * output) {
    
    arma::mat A(9,9); 
    
    if (input->is_bounds_query()) {
        input->dim[0].min = output->dim[0].min;
        input->dim[0].extent = output->dim[0].extent;
        input->dim[1].min = 0;
        input->dim[1].extent = 4;
        input->dim[2].min = 0;
        input->dim[2].extent = 4;
    } else {
        for (int n = 0; n < output->dim[0].extent; n++) {
            A.fill(0.0);
	        A(8,8)=1;
            for (int i = 0; i < 4; i++) {
                int coord_x1[3] = {n, i, 0};
                int coord_y1[3] = {n, i, 1};
                int coord_x2[3] = {n, i, 2};
                int coord_y2[3] = {n, i, 3};

                int x1 = *(int*)input->address_of(coord_x1);
                int y1 = *(int*)input->address_of(coord_y1);
                int x2 = *(int*)input->address_of(coord_x2);
                int y2 = *(int*)input->address_of(coord_y2);

                A(2*i, 0) =     x1;  A(2*i+1, 0) =    0.f;
                A(2*i, 1) =     y1;  A(2*i+1, 1) =    0.f;
                A(2*i, 2) =    1.f;  A(2*i+1, 2) =    0.f;
                A(2*i, 3) =    0.f;  A(2*i+1, 3) =     x1;
                A(2*i, 4) =    0.f;  A(2*i+1, 4) =     y1;
                A(2*i, 5) =    0.f;  A(2*i+1, 5) =    1.f;
                A(2*i, 6) = -x1*x2;  A(2*i+1, 6) = -x1*y2;
                A(2*i, 7) = -y1*x2;  A(2*i+1, 7) = -y1*y2;
                A(2*i, 8) =    -x2;  A(2*i+1, 8) =    -y2;
            }

            arma::vec B(9);
            B.fill(0);
            B(8) = 1;
            arma::vec X = arma::solve(A, B);
            arma::mat H(3, 3);
            H<<X(0)<<X(1)<<X(2)<<arma::endr
	         <<X(3)<<X(4)<<X(5)<<arma::endr
	         <<X(6)<<X(7)<<X(8)<<arma::endr;

            for (int k = 0; k < 3; k++) {
                for (int j = 0; j < 3; j++) {
                    int coord_out1[3] = {n, j, k};
                    *(float*)output->address_of(coord_out1) = H(k, j);
                }
            }
        }
        output->set_host_dirty();
    }

    return 0;
}

// Calculate inverse homography matrix
extern "C" DLLEXPORT int inverse_homography(halide_buffer_t * input, halide_buffer_t * output) {
    
    if (input->is_bounds_query()) {
        input->dim[0].min = output->dim[0].min;
        input->dim[0].extent = output->dim[0].extent;
        input->dim[1].min = output->dim[1].min;
        input->dim[1].extent = output->dim[1].extent;
    } else {
        
        arma::mat H(3, 3);

        for (int i = 0; i < output->dim[0].extent; i++) {
            for (int j = 0; j < output->dim[0].extent; j++) {
                    int coord[2] = {j ,i};
                    H(i, j) = *(float*)input->address_of(coord);
            }
        }

        arma::mat H1 = arma::inv(H);

        for (int i = 0; i < output->dim[0].extent; i++) {
            for (int j = 0; j < output->dim[0].extent; j++) {
                    int coord_out[2] = {j ,i};
                    *(float*)output->address_of(coord_out) = H1(i, j);
            }
        }
    }
    return 0;
}

Func compute_transformed_bbox(int width, int height, Func H) {
    
    Func prod_edge("dot_product_edge");
    Func output("output_compute_transofrmed_bbox");

    Var n("number_of_edge"), e("edge"), c, i, j;

    Func H1;
    H1(i, j) = f32(H(i, j));
    // calucate inverse homography matrix
    std::vector<ExternFuncArgument> args(1);
    args[0] = H1;
    Func inv_H("inverse_homography_matrix");
    inv_H.define_extern("inverse_homography", args, Float(32), {i, j});

    Buffer<int> edge(4, 2);
    edge(0, 0) =       0; edge(0, 1) =        0;
    edge(1, 0) = width-1; edge(1, 1) =        0;
    edge(2, 0) =       0; edge(2, 1) = height-1;
    edge(3, 0) = width-1; edge(3, 1) = height-1;

    prod_edge(n, e) = floor((inv_H(0, e) * edge(n, 0) + inv_H(1, e) * edge(n, 1) + inv_H(2, e)) /
                            (inv_H(0, 2) * edge(n, 0) + inv_H(1, 2) * edge(n, 1) + inv_H(2, 2)));
    
    RDom rn(0, 4);
    Expr xmin = min(minimum(prod_edge(rn, 0)), edge(0, 0));
    Expr xmax = max(maximum(prod_edge(rn, 0)), edge(3, 0));
    Expr ymin = min(minimum(prod_edge(rn, 1)), edge(0, 1));
    Expr ymax = max(maximum(prod_edge(rn, 1)), edge(3, 1));

    output(c) = cast<int>(select(c==0, xmin, c==1, xmax, c==2, ymin, ymax));

    // Schedule
    H1.compute_root();
    inv_H.compute_root().bound(i, 0, 3).bound(j, 0, 3);
    prod_edge.compute_root();
    output.compute_root().bound(c, 0, 4);

    return output;
}

Func compute_bbox(Func box, Func inv_H) {
    
    Func prod_edge("dot_product_edge");
    Func output("output_compute_transofrmed_bbox");

    Var n("number_of_edge"), e("edge"), c, i, j;

    Func edge;
    edge(n, e) = 0;
    edge(0, 0) = box(0); edge(0, 1) = box(2);
    edge(1, 0) = box(1); edge(1, 1) = box(2);
    edge(2, 0) = box(0); edge(2, 1) = box(3);
    edge(3, 0) = box(1); edge(3, 1) = box(3);

    prod_edge(n, e) = floor((inv_H(0, e) * edge(n, 0) + inv_H(1, e) * edge(n, 1) + inv_H(2, e)) /
                                  (inv_H(0, 2) * edge(n, 0) + inv_H(1, 2) * edge(n, 1) + inv_H(2, 2)));

    RDom rn(0, 4);
    Expr xmin = minimum(prod_edge(rn, 0));
    Expr xmax = maximum(prod_edge(rn, 0));
    Expr ymin = minimum(prod_edge(rn, 1));
    Expr ymax = maximum(prod_edge(rn, 1));

    output(c) = cast<int>(select(c==0, xmin, c==1, xmax, c==2, ymin, ymax));

    // Schedule
    edge.bound(n, 0, 4).bound(e, 0, 2);
    prod_edge.compute_root();
    output.compute_root().bound(c, 0, 4);

    return output;
}

Func bbox_union(std::vector<Func> cb_box) {
    
    int n = cb_box.size();
    Func B("Func_combine_box");
    Func output("output_bbox_union");
    Var m, c;

    B(m, c) = 0;
    for (int i = 0; i < n; i++) {
        B(i, c) = cb_box[i](c);
    }
    
    RDom r(0, n);
    Expr xmin = argmin(B(r, 0))[1];
    Expr xmax = argmax(B(r, 1))[1];
    Expr ymin = argmin(B(r, 2))[1];
    Expr ymax = argmax(B(r, 3))[1];

    output(c) = select(c==0, xmin, c==1, xmax, c==2, ymin, ymax);

    // Schedule
    output.compute_root().bound(c, 0, 4);

    return output;
}

Func apply_homography(Func black, Func im, Func H, int w, int h, Func bbox) {

    Var x("x"), y("y"), c("c"), e("e");
    Func output("output_homograpy_image");
    Func prod("dot_product_of_all_xy");
    Func prodf("dot_product_of_all_xy_floor");
    Func prodr("dot_product_of_all_xy_roof");
    Func weightf("weight_floor");
    Func weightr("weight_roof");

    prod(x, y, e) = (H(0, e) * x + H(1, e) * y + H(2, e)) /
                    (H(0, 2) * x + H(1, 2) * y + H(2, 2));
    
    prodf(x, y, e) = select(e==0, clamp(cast<int>(floor(prod(x, y, e))), 0, w-1),
                                  clamp(cast<int>(floor(prod(x, y, e))), 0, h-1));
    prodr(x, y, e) = select(e==0, clamp(cast<int>(prodf(x, y, e) + 1), 0, w-1),
                                  clamp(cast<int>(prodf(x, y, e) + 1), 0, h-1));
    
    weightf(x, y, e) = 1.f - (prod(x, y, e) - prodf(x, y, e));
    weightr(x, y, e) = 1.f - weightf(x, y, e);

    Expr out_floor = (prod(x, y, 0) < 0 || prod(x, y, 0) > w ||
                      prod(x, y, 1) < 0 || prod(x, y, 1) > h);
    Expr out_round = (prod(x, y, 0) < -1  || prod(x, y, 0) > w-1 ||
                      prod(x, y, 1) < -1  || prod(x, y, 1) > h-1);
    
    if (im.dimensions() != 3) {
        output(x, y) = select(out_floor,
                               cast<float>(black(x, y, 0)),
                               cast<float>(im(clamp(prodf(x, y, 0), 0, w-1),
                                              clamp(prodf(x, y, 1), 0, h-1))));
    } else {
        output(x, y, c) = select(out_floor && out_round, cast<float>(black(x, y, c)),
            cast<float>(im(prodf(x, y, 0), prodf(x, y, 1), c)) * weightf(x, y, 0) * weightf(x, y, 1) +
            cast<float>(im(prodr(x, y, 0), prodf(x, y, 1), c)) * weightr(x, y, 0) * weightf(x, y, 1) +
            cast<float>(im(prodf(x, y, 0), prodr(x, y, 1), c)) * weightf(x, y, 0) * weightr(x, y, 1) +
            cast<float>(im(prodr(x, y, 0), prodr(x, y, 1), c)) * weightr(x, y, 0) * weightr(x, y, 1));

    }

    // Schedule
    prod.compute_root().parallel(y).vectorize(x, 16);
    prodf.compute_root().parallel(y).vectorize(x, 16);
    prodr.compute_root().parallel(y).vectorize(x, 16);
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func apply_finalH(Func im, Func H, int w, int h) {

    Var x("x"), y("y"), c("c"), e("e");
    Func output("output_homograpy_image");
    Func prod("dot_product_of_all_xy");
    Func prodf("dot_product_of_all_xy_floor");
    Func prodr("dot_product_of_all_xy_roof");
    Func weightf("weight_floor");
    Func weightr("weight_roof");

    prod(x, y, e) = (H(0, e) * x + H(1, e) * y + H(2, e)) /
                    (H(0, 2) * x + H(1, 2) * y + H(2, 2));
    
    prodf(x, y, e) = select(e==0, clamp(cast<int>(floor(prod(x, y, e))), 0, w-1),
                                  clamp(cast<int>(floor(prod(x, y, e))), 0, h-1));
    prodr(x, y, e) = select(e==0, clamp(cast<int>(prodf(x, y, e) + 1), 0, w-1),
                                  clamp(cast<int>(prodf(x, y, e) + 1), 0, h-1));
    
    weightf(x, y, e) = 1.f - (prod(x, y, e) - prodf(x, y, e));
    weightr(x, y, e) = 1.f - weightf(x, y, e);

    Expr out_floor = (prod(x, y, 0) < 0 || prod(x, y, 0) > w ||
                      prod(x, y, 1) < 0 || prod(x, y, 1) > h);
    Expr out_round = (prod(x, y, 0) < -1  || prod(x, y, 0) > w-1 ||
                      prod(x, y, 1) < -1  || prod(x, y, 1) > h-1);
    
    if (im.dimensions() != 3) {
        output(x, y) = select(out_floor,
                               cast<float>(0),
                               cast<float>(im(clamp(prodf(x, y, 0), 0, w-1),
                                              clamp(prodf(x, y, 1), 0, h-1))));
    } else {
        output(x, y, c) = select(out_floor && out_round, cast<float>(0),
            cast<float>(im(prodf(x, y, 0), prodf(x, y, 1), c)) * weightf(x, y, 0) * weightf(x, y, 1) +
            cast<float>(im(prodr(x, y, 0), prodf(x, y, 1), c)) * weightr(x, y, 0) * weightf(x, y, 1) +
            cast<float>(im(prodf(x, y, 0), prodr(x, y, 1), c)) * weightf(x, y, 0) * weightr(x, y, 1) +
            cast<float>(im(prodr(x, y, 0), prodr(x, y, 1), c)) * weightr(x, y, 0) * weightr(x, y, 1));
    }

    

    // Schedule
    prod.compute_at(output, y).parallel(y).vectorize(x, 16);
    prodf.compute_at(output, y).parallel(y).vectorize(x, 16);
    prodr.compute_at(output, y).parallel(y).vectorize(x, 16);
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func color_correct(Func im1, Func im2, Expr w1, Expr h1, Expr w2, Expr h2, Func weight1, Func weight2) {
    Func mean1("mean1");
    Func mean2("mean2");
    Func std1("std1");
    Func std2("std2");
    Func num("number_of_points");
    Var x, y, c;

    RDom r1(0, w2*2, 0, h2*2);
    r1.where(weight1(r1.x, r1.y) > 0 && weight2(r1.x, r1.y) > 0);
    num(c) = sum(r1.x/r1.x);

    mean1(c) = sum(f32(im1(r1.x, r1.y, c))) / num(c);
    std1(c) = sqrt(sum(pow(f32(im1(r1.x, r1.y, c)) - mean1(c),2)) / num(c));

    mean2(c) = sum(f32(im2(r1.x, r1.y, c))) / num(c);
    std2(c) = sqrt(sum(pow(f32(im2(r1.x, r1.y, c)) - mean2(c),2)) / num(c));

    Func output("output_color_correct");
    output(x, y, c) = select(weight2(x, y) > 0,
                        f32((im2(x, y, c) - mean2(c)) * std1(c) / std2(c) + mean1(c)), f32(0));

    // Schedule
    num.compute_root(); 
    mean1.compute_root();
    mean2.compute_root();
    std1.compute_root();
    std2.compute_root();
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func color_correction(Func im1, Func im2, Func weight1, Func weight2, int max_w, int max_h) {
    Func mean1("mean1");
    Func mean2("mean2");
    Func std1("std1");
    Func std2("std2");
    Func num("number_of_points");
    Var x, y, c;

    RDom r1(0, max_w, 0, max_h);
    r1.where(weight1(r1.x, r1.y) > 0 && weight2(r1.x, r1.y) > 0);
    num(c) = sum(r1.x/r1.x);

    mean1(c) = sum(f32(im1(r1.x, r1.y, c))) / num(c);
    std1(c) = sqrt(sum(pow(f32(im1(r1.x, r1.y, c)) - mean1(c),2)) / num(c));

    mean2(c) = sum(f32(im2(r1.x, r1.y, c))) / num(c);
    std2(c) = sqrt(sum(pow(f32(im2(r1.x, r1.y, c)) - mean2(c),2)) / num(c));

    Func output("output_color_correct");
    output(x, y, c) = select(weight2(x, y) > 0,
                        f32((im2(x, y, c) - mean2(c)) * std1(c) / std2(c) + mean1(c)), f32(0));

    // Schedule
    num.compute_root(); 
    mean1.compute_root();
    mean2.compute_root();
    std1.compute_root();
    std2.compute_root();
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}


Func color_match(Func im1, Func im2, Func weight1, Func weight2, int width, int height, Func inv_H1, Func inv_H2, int wmax, int hmax, Func box) {
    Func mean1("mean1");
    Func mean2("mean2");
    Func std1("std1");
    Func std2("std2");
    Func num1("number_of_points");
    Func num2("number_of_points");
    Var x, y, c, e;

    Func inv_P1("inv_Point");
    inv_P1(x, y, e) = i32(floor((inv_H1(0, e) * x + inv_H1(1, e) * y + inv_H1(2, e)) /
                               (inv_H1(0, 2) * x + inv_H1(1, 2) * y + inv_H1(2, 2))));

    Func bound_inv_x1("bound_inverse_xpoint");
    Func bound_inv_y1("bound_inverse_ypoint");
    bound_inv_x1(x, y) = clamp(inv_P1(x, y, 0), 0, wmax);
    bound_inv_y1(x, y) = clamp(inv_P1(x, y, 1), 0, hmax);

    RDom r1(0, width, 0, height);
    r1.where(weight1(bound_inv_x1(r1.x, r1.y), bound_inv_y1(r1.x, r1.y)) > 0 &&
             weight2(bound_inv_x1(r1.x, r1.y), bound_inv_y1(r1.x, r1.y)) > 0);
    
    num1(c) = sum(r1.x/r1.x);

    mean1(c) = sum(f32(im1(r1.x, r1.y, c))) / num1(c);
    std1(c) = sqrt(sum(pow(f32(im1(r1.x, r1.y, c)) - mean1(c),2)) / num1(c));

    Func inv_P2("inv_Point");
    inv_P2(x, y, e) = i32(floor((inv_H2(0, e) * x + inv_H2(1, e) * y + inv_H2(2, e)) /
                               (inv_H2(0, 2) * x + inv_H2(1, 2) * y + inv_H2(2, 2))));

    Func bound_inv_x2("bound_inverse_xpoint");
    Func bound_inv_y2("bound_inverse_ypoint");
    bound_inv_x2(x, y) = clamp(inv_P2(x, y, 0), 0, wmax);
    bound_inv_y2(x, y) = clamp(inv_P2(x, y, 1), 0, hmax);

    RDom r2(0, width, 0, height);
    r2.where(weight1(bound_inv_x2(r2.x, r2.y), bound_inv_y2(r2.x, r2.y)) > 0 &&
             weight2(bound_inv_x2(r2.x, r2.y), bound_inv_y2(r2.x, r2.y)) > 0);

    num2(c) = sum(r2.x/r2.x);

    mean2(c) = sum(f32(im2(r2.x, r2.y, c))) / num2(c);
    std2(c) = sqrt(sum(pow(f32(im2(r2.x, r2.y, c)) - mean2(c),2)) / num2(c));

    Func output("output_color_correct");
    output(x, y, c) = f32((im2(x, y, c) - mean2(c)) * std1(c) / std2(c) + mean1(c));
    
    // Schedule
    mean1.compute_root();
    mean2.compute_root();
    std1.compute_root();
    std2.compute_root();
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func prodH(Func H1, Func H2) {
    
    Func output;
    Var j, k;
    output(j, k) = H1(0, k) * H2(j, 0) +
                   H1(1, k) * H2(j, 1) +
                   H1(2, k) * H2(j, 2);

    // Schedule
    output.compute_root().bound(j, 0, 3).bound(k, 0, 3);

    return output;
}

Func prodP(Func H, Expr x, Expr y) {
    
    Func output("product_point");
    Var e;
    output(e) = (H(0, e) * x + H(1, e) * y + H(2, e)) /
                (H(0, 2) * x + H(1, 2) * y + H(2, 2));

    // Schedule
    output.compute_root().bound(e, 0, 3);

    return output;
}

Func onesH() {
    
    Func output("ones_matrix");
    Var x, y;
   
    output(x, y) = 0.f;
    output(0, 0) = 1.f;
    output(1, 1) = 1.f;
    output(2, 2) = 1.f;

    // Schedule
    output.compute_root().bound(x, 0, 3).bound(y, 0, 3);

    return output;
}

Func translate_correspondence(Func correspondence, Func inv_H) {

    Func output("output_translate_correspondence");
    Var tx, ty, c, e;
    Expr x = f32(correspondence(tx, ty, 0));
    Expr y = f32(correspondence(tx, ty, 1));
    
    Func transp; 
    transp(tx, ty, e) = cast<int>(floor((inv_H(0, e) * x + inv_H(1, e) * y + inv_H(2, e)) /
                                        (inv_H(0, 2) * x + inv_H(1, 2) * y + inv_H(2, 2))));

    output(tx, ty, c) = select(correspondence(tx, ty, 0) != -1,
                                select(c == 0, transp(tx, ty, 0),
                                       c == 1, transp(tx, ty, 1),
                                       c == 2, correspondence(tx, ty, 2),
                                               correspondence(tx, ty, 3)),
                                -1);
 
    // Schedule
    output.compute_root().reorder(c, tx, ty).parallel(ty).vectorize(tx);

    return output;
}

Func inverse_homography(Func H) {

    Func H1;
    Var i, j;
    H1(i, j) = f32(H(i, j));

    // calucate inverse homography matrix
    std::vector<ExternFuncArgument> args(1);
    args[0] = H1;
    Func inv_H("inverse_homography_matrix");
    inv_H.define_extern("inverse_homography", args, Float(32), {i, j});
    
    // Schedule
    H1.compute_root().bound(i,0,3).bound(j,0,3);
    inv_H.compute_root().bound(i,0,3).bound(j,0,3);

    return inv_H;
}

Func downscale(Func im, int n) {
    
    Func output("output_downscale");
    Var x, y, c;
    output(x, y, c) = im(n*x, n*y, c);

    // Schedule
    output.compute_root().parallel(y);

    return output;
}

Func calculate_weight(int w, int h) {

    Func weight("weight");
    Var x, y;
    weight(x, y) = 0.25f * 
                select(x < w/2, cast<float>(x)/w, cast<float>(w-x)/w) *
                select(y < h/2, cast<float>(y)/h, cast<float>(h-y)/h);

    weight.compute_root().parallel(y).vectorize(x, 16);

    return weight;
}

Func calculate_highfrequency(Func im, Func low) {

    Func high_frequency;
    Var x, y, c;
    high_frequency(x, y, c) = f32(im(x, y, c) - low(x, y, c));

    return high_frequency;
}

Func stitch_high(Func high1, Func high2, Func transweight1, Func transweight2) {

    Func final_high("final_high");
    Func output;
    Var x, y, c;

    float pi = 3.1415926f;
    float n = 2.f;
    Func weight12, weight21;
    weight12(x, y) = 0.5f - 0.5f*cos(pi*transweight1(x, y) / (transweight2(x, y) * n));
    weight21(x, y) = 0.5f - 0.5f*cos(pi*transweight2(x, y) / (transweight1(x, y) * n));
    
    Expr con1 = transweight1(x, y) > transweight2(x, y);
    Expr con3 = transweight1(x, y) / transweight2(x, y) < n;
    Expr con2 = transweight2(x, y) > transweight1(x, y);
    Expr con4 = transweight2(x, y) / transweight1(x, y) < n;  
    
    final_high(x, y, c) = select(transweight1(x, y) > 0,
                            select(transweight2(x, y) > 0,
                                select( con1 && con3,    weight12(x, y)  * high1(x, y, c)+ 
                                                      (1-weight12(x, y)) * high2(x, y, c),
                                        con2 && con4, (1-weight21(x, y)) * high1(x, y, c)+
                                                         weight21(x, y)  * high2(x, y, c),
                                        con1 && !con3, high1(x, y, c), high2(x, y, c)),
                            high1(x, y, c)),
                        high2(x, y, c));
    
    // Schedule
    weight12.compute_root();
    weight21.compute_root();
    final_high.compute_root().parallel(y).vectorize(x, 16);

    return final_high;
}

Func stitch_low(Func low1, Func low2, Func transweight1, Func transweight2) {

    Func final_low("final_low");
    Func output("output_stitch_low");
    Var x, y, c;

    float pi = 3.1415926f;
    float n = 2.f;
    Func weight12, weight21;
    weight12(x, y) = 0.5f - 0.5f*cos(pi*transweight1(x, y) / (transweight2(x, y) * n));
    weight21(x, y) = 0.5f - 0.5f*cos(pi*transweight2(x, y) / (transweight1(x, y) * n));
    
    Expr con1 = transweight1(x, y) > transweight2(x, y);
    Expr con3 = transweight1(x, y) / transweight2(x, y) < n;
    Expr con2 = transweight2(x, y) > transweight1(x, y);
    Expr con4 = transweight2(x, y) / transweight1(x, y) < n; 
 
    final_low(x, y, c) = select(transweight1(x, y) > 0,
                            select(transweight2(x, y) > 0,
                                select( con1 && con3,    weight12(x, y)  * low1(x, y, c)+ 
                                                      (1-weight12(x, y)) * low2(x, y, c),
                                        con2 && con4, (1-weight21(x, y)) * low1(x, y, c)+
                                                         weight21(x, y)  * low2(x, y, c),
                                        con1 && !con3, low1(x, y, c), low2(x, y, c)),
                                    low1(x, y, c)),
                            low2(x, y, c));
    
    // Schedule
    weight12.compute_root();
    weight21.compute_root();
    final_low.compute_root().parallel(y).vectorize(x, 16);

    return final_low;
}

Func to_uint8(Func im) {
    
    Func output;
    Var x, y, c;
    output(x, y, c) = u8(im(x, y, c));

    // Schedule
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;
}

Func combine_weight(Func weight1, Func weight2) {

    Func output;
    Var x, y, c;
    if (weight1.dimensions() != 3) {
        output(x, y) = weight1(x, y) + weight2(x, y);
    } else {
        output(x, y, c) = weight1(x, y, c) + weight2(x, y, c);
    }


    // Schedule
    output.compute_root().parallel(y).vectorize(x, 16);

    return output;  
}