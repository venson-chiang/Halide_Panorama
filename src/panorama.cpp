#include "panorama.h"
#include "utils.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

// Find Harris corners
Func harris_corners(Func im, Expr width, Expr height, int max_diam, float k, double sigma,
                    double factor, double boundary_size) {
    
    Func corner_response("CornerResponse");
    Func maximum_window_x("maxWindowX");
    Func maximum_window("maxWindow");
    Func show_corners("show_corners");
    Func corner_xy("record_corner_xy");
    Func output("output_harris_corners");

    Var x, y, c;
    RDom r(-(max_diam-1), 2*max_diam-1);

    // get the tensor
    Func luminance_blurred = blur_gaussian(im, sigma, "luminance_blurred");
    Func tensor = calculate_tensor(luminance_blurred, width, height, sigma, factor);

    // calculate corner response
    corner_response(x, y) = tensor(x, y, 0) * tensor(x, y, 2) - pow(tensor(x, y, 1), 2)
                        -k * (pow((tensor(x, y, 0) + tensor(x, y, 2)), 2));

    //find maximum value in each window
    maximum_window_x(x, y) = maximum(corner_response(x+r, y));
    maximum_window(x, y) = maximum(maximum_window_x(x, y+r));

    // find corners (local max)
    show_corners(x, y) = select((corner_response(x, y) == maximum_window(x, y)) 
                                && corner_response(x, y) > 0.f
                                ,cast<uint8_t>(255)
                                ,cast<uint8_t>(0));

    // Record corner xy position
    Var tx, ty, num_c, c_xy;
    RDom t(0, max_diam , 0, max_diam);
    corner_xy(tx, ty) = argmax(show_corners(max_diam*tx+t.x, max_diam*ty+t.y));

    // if argmax == 255, output it's xy location, otherwise output it -1 -1
    output(tx, ty, c_xy) = select(corner_xy(tx, ty)[2] == 255,
                                select(c_xy == 0, max_diam * tx + corner_xy(tx, ty)[0], 
                                                  max_diam * ty + corner_xy(tx, ty)[1]),
                                -1);
    
    // Schedule
    corner_response.compute_root().parallel(y).vectorize(x, 16);
    maximum_window_x.compute_root().parallel(y).vectorize(x, 16);
    maximum_window.compute_root().parallel(y).vectorize(x, 16);
    show_corners.compute_root().parallel(y).vectorize(x, 16);
    corner_xy.compute_root();
    output.reorder(c_xy, tx, ty).compute_root();

    return output;
}

// Compute normalized feature descriptor around harris corners
Func compute_feature(Func im, Expr width, Expr height, Func corners, 
                     float sigma_blur_descriptor, int radius_descriptor) {

    Func descriptor("descriptor");
    Func mean("mean");
    Func st_dev("st_dev");
    Func output("output_normalized_feature_descriptor");

    Var tx, ty, x, y, c;
    float area = (radius_descriptor * 2.f + 1.f) * (radius_descriptor * 2.f + 1.f);
    RDom r(0, radius_descriptor*2+1, 0, radius_descriptor*2+1);

    // blur luminace im
    Func luminance_blurred = blur_gaussian(im, sigma_blur_descriptor, "feature_blur");

    // calculate descriptor
    descriptor(tx, ty, x, y) = select(corners(tx, ty, 0) != -1,
                                luminance_blurred(
                                clamp(corners(tx, ty, 0)+x-radius_descriptor, 0, width),
                                clamp(corners(tx, ty, 1)+y-radius_descriptor, 0, height)),
                                0);

    // calculate mean and standard deviation
    mean(tx, ty) = sum(descriptor(tx, ty, r.x, r.y)) / area;
    st_dev(tx, ty) = sqrt(sum(pow(descriptor(tx, ty, r.x, r.y) - mean(tx, ty), 2)) / area);

    // normalized_descriptor
    output(tx, ty, x, y) = select(corners(tx, ty, 0) != -1,
                                 (mean(tx, ty) - descriptor(tx, ty, x, y))/ st_dev(tx, ty),
                                  0);
    
    // Schedule
    descriptor.compute_root();
    mean.compute_root();
    st_dev.compute_root();
    output.reorder(x, y, tx, ty).compute_root().parallel(ty);

    return output;
}

// Find corresponedences of two features, return best matching points
Func find_correspondences(Func feature_1, Func feature_2, Func corner_1, Func corner_2, int max_diam,
                    int num_tile_x, int num_tile_y, int radius_descriptor, float threshold) {

    Func leastSquares("LeastSquare_error");
    Func min_pair1("minimum_pair");
    Func leastSquares2("2nd leastSquares error");
    Func min_pair2("second_min_pair");
    Func output("output_correspondent_corners");

    Var tx1, ty1, tx2, ty2, x, y, c;
    RDom r(0, radius_descriptor*2+1, 0, radius_descriptor*2+1);

    // calculate sqaure error between any two descroptors
    leastSquares(tx1, ty1, tx2, ty2) = select(corner_1(tx1, ty1, 0) != -1,
                                              sum(pow(feature_1(tx1, ty1, r.x, r.y) - 
                                                      feature_2(tx2, ty2, r.x, r.y), 2)),
                                              INFINITY);

    // Find smallest pair between feature_1 and feature_2
    RDom t(0, num_tile_x, 0, num_tile_y);
    t.where(corner_2(t.x, t.y, 0) != -1);
    
    min_pair1(tx1, ty1) = argmin(leastSquares(tx1, ty1, t.x, t.y));
    
    // Find second small pair between feature_1 and feature_2
    leastSquares2(tx1, ty1, tx2, ty2) = select(
                            tx2 != min_pair1(tx1, ty1)[0] && ty2 != min_pair1(tx1, ty1)[1],
                            leastSquares(tx1, ty1, tx2, ty2), INFINITY);

    min_pair2(tx1, ty1) = argmin(leastSquares2(tx1, ty1, t.x, t.y));

    // output if difference of minimum pair is zero 
    // or the difference ratio of 2nd / 1st min pair is larger than threshold 
    output(tx1, ty1, c) = select((corner_1(tx1, ty1, 0) != -1) &&
                                 (min_pair1(tx1, ty1)[2] == 0 || 
                                  min_pair2(tx1, ty1)[2]/min_pair1(tx1, ty1)[2] > threshold ),
                                select(c == 0, corner_1(tx1, ty1, 0),
                                       c == 1, corner_1(tx1, ty1, 1),
                                       c == 2, corner_2(clamp(min_pair1(tx1, ty1)[0],0,959), clamp(min_pair1(tx1, ty1)[1],0,959), 0),
                                               corner_2(clamp(min_pair1(tx1, ty1)[0],0,959), clamp(min_pair1(tx1, ty1)[1],0,959), 1)),
                                -1);

    // Schedule
    leastSquares.reorder(tx2, ty2, tx1, ty1).compute_at(output, ty1).parallel(ty2).vectorize(tx2, 16);
    min_pair1.compute_at(output, ty1);
    leastSquares2.compute_at(output, ty1);
    min_pair2.compute_at(output, ty1);
    output.reorder(c, tx1, ty1).compute_root().parallel(ty1).vectorize(tx1, 16);

    return output;
}

// Apply ransanc to find homography matrix
Func ransac(Func correspondence, int num_tile_x, int num_tile_y, int Niter, float epsilon) {

    Func point_list("point_list");
    Func r_tx("random_tile_x");
    Func r_ty("random_tile_y");
    Func homography("homography_matrix");
    Func a_tx("all_valid_tile_x");
    Func a_ty("all_valid_tile_y");
    Func prod("products");
    Func dif("difference");
    Func num_inlier("number_of_inlier");
    Func output("output_homography_matrix");

    Var iter("iteration"), rt("random_tile"), c("correspondence"), j("H_dim0"), k("H_dim1");
    Var t("tile"), oj("output_dim0"), ok("output_dim1"), inv("inverse");

    // Find 4 random valid tiles from sorted correspondence pairs
    Func sorted_list = merge_sort(correspondence, num_tile_x, num_tile_y);
    Func valid_length = num_valid_corners(sorted_list, num_tile_x*num_tile_y);
    Func random_tiles = generate_4p(valid_length(0));
    
    // Find tile_x, tile_y from 4 random valid tiles
    r_tx(iter, rt) = clamp(sorted_list(clamp(random_tiles(iter, rt),0,num_tile_x*num_tile_y)) % num_tile_x, 0, num_tile_x);
    r_ty(iter, rt) = clamp(sorted_list(clamp(random_tiles(iter, rt),0,num_tile_x*num_tile_y)) / num_tile_x, 0, num_tile_y);
    
    // Find correspondent x, y coordinate from tile_x and tile_y
    point_list(iter, rt, c) = correspondence(r_tx(iter, rt), r_ty(iter, rt), c);

    // Use extern function to calculate linear algebra solution of homography matix
    std::vector<ExternFuncArgument> args(1);
    args[0] = point_list;
    homography.define_extern("calculate_homography", args, Float(32), {iter, j ,k});

    // Find tile_x, tile_y from all valid tiles
    a_tx(t) = clamp(sorted_list(t) % num_tile_x, 0, num_tile_x); 
    a_ty(t) = clamp(sorted_list(t) / num_tile_x, 0, num_tile_y); 

    // transform im2's points by H of every iteraton
    prod(iter, t, k) = floor(
                         (homography(iter, 0, k) * correspondence(a_tx(t), a_ty(t), 0) +
                          homography(iter, 1, k) * correspondence(a_tx(t), a_ty(t), 1) +
                          homography(iter, 2, k)) / 
                         (homography(iter, 0, 2) * correspondence(a_tx(t), a_ty(t), 0) +
                          homography(iter, 1, 2) * correspondence(a_tx(t), a_ty(t), 1) +
                          homography(iter, 2, 2)) 
                         );
    
    // calculate square error between correspondence points of im1 and transformed im2
    dif(iter, t) = pow(correspondence(a_tx(t), a_ty(t), 2) - prod(iter, t, 0), 2) +
                   pow(correspondence(a_tx(t), a_ty(t), 3) - prod(iter, t, 1), 2);
    
    // calcualte how many correspendent point's difference is smaller than epsilon
    RDom vt(0, num_tile_x*num_tile_y);
    vt.where(vt < valid_length(0));
    num_inlier(iter) = sum(select(dif(iter, vt) < epsilon*epsilon, 1, 0));

    // calculate max num_inlier index
    RDom ir(0, Niter);
    Tuple max_iter = argmax(num_inlier(ir));
    output(oj, ok) = homography(clamp(max_iter[0], 0, 10000), oj, ok);

    // Schedule
    point_list.reorder(c, rt ,iter).compute_root().parallel(iter);
    homography.compute_root().bound(j, 0, 3).bound(k, 0 ,3);
    prod.reorder(k, t, iter).compute_at(num_inlier, iter);
    dif.reorder(t, iter).compute_at(num_inlier, iter);
    num_inlier.compute_root().parallel(iter);
    output.compute_root();

    return output;
}

// Stitch 2 images
Func stitch(Func im1, Func im2, int w1, int h1, int w2, int h2, Func H, Func bbox) {

    Func weight1("weight1"), weight2("weight2");
    Var x("x1"), y("xy"), c("xc"), i, j;

    Expr stich_width = bbox(1) - bbox(0) + 1;
    Expr stich_height = bbox(3) - bbox(2) + 1;

    Func translate("translate_matrix");
    translate(i, j) = 0.f;
    translate(0, 0) = 1.f; translate(2, 0) = f32(bbox(0)); 
    translate(1, 1) = 1.f; translate(2, 1) = f32(bbox(2));
    translate(2, 2) = 1.f; 

    Func black("black");
    black(x, y, c) = f32(0);
    

    // calculate weights
    weight1(x, y) = 0.25f * 
                    select(x < w1/2, cast<float>(x)/w1, cast<float>(w1-x)/w1) *
                    select(y < h1/2, cast<float>(y)/h1, cast<float>(h1-y)/h1);
    weight2(x, y) = 0.25f * 
                    select(x < w2/2, cast<float>(x)/w2, cast<float>(w2-x)/w2) *
                    select(y < h2/2, cast<float>(y)/h2, cast<float>(h2-y)/h2);
    
    // translate the first image, need translate matrix
    Func transformed1 = apply_homography(black, im1, translate, w1, h1, bbox);
    Func transformed_weight1 = apply_homography(black, weight1, translate, w1, h1, bbox);

    // translate the second image, need use translate_matrix * homography matrix
    Func H2("translate*homography_matrix");
    H2(i, j) = H(0, j) * translate(i, 0) +
               H(1, j) * translate(i, 1) +
               H(2, j) * translate(i, 2);

    Func transformed2_temp = apply_homography(black, im2, H2, w2, h2, bbox);
    Func transformed_weight2 = apply_homography(black, weight2, H2, w2, h2, bbox);

    Func transformed2 = color_correct(transformed1, transformed2_temp, stich_width, stich_height, 
                                    w2, h2, transformed_weight1, transformed_weight2);

    // calculate high and low frequency images
    Func low_frequency1 = apply_homography(black, blur_gaussian(im1, 15, "low_frequency1"), translate, w1, h1, bbox);
    Func high_frequency1("high_frquency1");
    high_frequency1(x, y, c) = f32(transformed1(x, y, c) - low_frequency1(x, y, c));

    Func low_frequency2_temp = apply_homography(black, blur_gaussian(im2, 15, "low_frequency1"), H2, w2, h2, bbox);

    Func low_frequency2 = color_correct(low_frequency1, low_frequency2_temp
                                        ,stich_width, stich_height, w2, h2, transformed_weight1, transformed_weight2);

    Func high_frequency2("high_frquency2");
    high_frequency2(x, y, c) = f32(transformed2(x, y, c) - low_frequency2(x, y, c));

    Func diff_high12("diff_high"), diff_high21("diff_high21");
    diff_high12(x, y, c) = select(transformed_weight1(x, y) > 0 && transformed_weight2(x, y) > 0,
                                select(transformed1(x, y, c) - transformed2(x, y, c) > 30.f, 1.f, 
                                       transformed2(x, y, c) - transformed1(x, y, c) > 30.f, 2.f,
                                                                                             3.f), 0.f);

    diff_high21(x, y, c) = select(transformed_weight1(x, y) > 0 && transformed_weight2(x, y) > 0
                                && high_frequency2(x, y, c) - high_frequency1(x, y, c) > 5.f,
                                0.f , 1.f);

    // calculate final high and low frequency
    Func final_high("final_high");
    float pi = 3.1415926f;
    float n = 2.f;
    Func weight12, weight21;
    weight12(x, y) = 0.5f - 0.5f*cos(pi*transformed_weight1(x, y) / (transformed_weight2(x, y) * n));
    weight21(x, y) = 0.5f - 0.5f*cos(pi*transformed_weight2(x, y) / (transformed_weight1(x, y) * n));

    Expr con1 = transformed_weight1(x, y) > transformed_weight2(x, y);
    Expr con3 = transformed_weight1(x, y) / transformed_weight2(x, y) < n;

    Expr con2 = transformed_weight2(x, y) > transformed_weight1(x, y);
    Expr con4 = transformed_weight2(x, y) / transformed_weight1(x, y) < n;                   

    Func final_trans("final_trans");
    final_trans(x, y, c) = select(transformed_weight1(x, y) > transformed_weight2(x, y),
                                  transformed1(x, y, c), transformed2(x, y, c));

    Func difweight("diff_weight");
    difweight(x, y, c) = 0.5f * abs(transformed1(x, y, c) - transformed2(x, y, c)) / 256.f;

    Func weight12_1, weight12_2, weight21_1, weight21_2;

    weight12_1(x, y, c) = weight12(x, y) * difweight(x, y, c) / 
                         (weight12(x, y) * difweight(x, y, c) + (1-weight12(x, y)) * (1-difweight(x, y, c)));
    
    weight12_2(x, y, c) = 1.f - weight12_1(x, y, c);

    weight21_2(x, y, c) = weight21(x, y) * difweight(x, y, c) / 
                         (weight21(x, y) * difweight(x, y, c) + (1-weight21(x, y)) * (1-difweight(x, y, c)));

    weight21_1(x, y, c) = 1.f - weight21_2(x, y, c);
   
    final_high(x, y, c) = select(transformed_weight1(x, y) > 0,
                            select(transformed_weight2(x, y) > 0,
                                select( con1 && con3,    weight12(x, y)  * high_frequency1(x, y, c)+ 
                                                      (1-weight12(x, y)) * high_frequency2(x, y, c),
                                        con2 && con4, (1-weight21(x, y)) * high_frequency1(x, y, c)+
                                                         weight21(x, y)  * high_frequency2(x, y, c),
                                        con1 && !con3, high_frequency1(x, y, c), high_frequency2(x, y, c)),
                            high_frequency1(x, y, c)),
                        high_frequency2(x, y, c));
    
    Func final_low("final_low");
    final_low(x, y, c) = select(transformed_weight1(x, y) > 0,
                            select(transformed_weight2(x, y) > 0,
                                select( con1 && con3,    weight12(x, y)  * low_frequency1(x, y, c)+ 
                                                      (1-weight12(x, y)) * low_frequency2(x, y, c),
                                        con2 && con4, (1-weight21(x, y)) * low_frequency1(x, y, c)+
                                                         weight21(x, y)  * low_frequency2(x, y, c),
                                        con1 && !con3, low_frequency1(x, y, c), low_frequency2(x, y, c)),
                                    low_frequency1(x, y, c)),
                            low_frequency2(x, y, c));

    Func overlap1("overlap1");
    overlap1(x, y, c) = select(transformed_weight1(x, y) > 0 && transformed_weight2(x, y) > 0,
                                transformed1(x, y, c), 0);
    
    Func output("output");
    output(x, y, c) = u8(final_high(x, y, c) + final_low(x, y, c));

    // Schedule
    translate.compute_root();
    black.compute_root();
    weight1.compute_root();
    weight2.compute_root();
    final_high.compute_root().parallel(y).vectorize(x, 16);
    final_low.compute_root().parallel(y).vectorize(x, 16);
    output.compute_root().parallel(y).vectorize(x, 16);
    
    return output;
}

// Visualize harris corners
Func visualize_harris(Func im, Func corner, int num_tile_x, int num_tile_y, int width, int height) {
    Func output;
    Var x, y, c, tx, ty;
    output(x, y, c) = u8(im(x, y, c));

    RDom r(0, num_tile_x, 0, num_tile_y);
    output(clamp(corner(r.x, r.y, 0),0, width-1), 
            clamp(corner(r.x, r.y, 1), 0, height-1), c) = select(c==0,u8(255), u8(0));

    return output;
}

// Visualize correspondent corners of image
Func visualize_correspondence(Func im, Func correspondence, int num_tile_x, int num_tile_y, int n, int width ,int height) {

    Func output;
    Var x, y, c, tx, ty;
    output(x, y, c) = u8(im(x, y, c));

    RDom r(0, num_tile_x, 0, num_tile_y);

    if (n == 0) {
    output(clamp(correspondence(r.x, r.y, 0),0, width-1), 
        clamp(correspondence(r.x, r.y, 1), 0, height-1), c) = select(c==0,u8(255), u8(0));
    } else {
    output(clamp(correspondence(r.x, r.y, 2),0, width-1), 
            clamp(correspondence(r.x, r.y, 3), 0, height-1), c) = select(c==0,u8(255), u8(0));
    }

    return output;
}