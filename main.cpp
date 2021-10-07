#include <Halide.h>
#include <halide_image_io.h>
#include <src/panorama.h>
#include <src/utils.h>
#define ARMA_WARN_LEVEL 1
#include <armadillo>

using namespace Halide;
using namespace Halide::ConciseCasts;
using namespace Halide::Tools;

int main(int argc, char** argv) {

    arma::arma_version ver;
    std::cout << "ARMA version: "<< ver.as_string() << std::endl;
    
    std::vector<Buffer<uint8_t>> im;
    for (int i = 2; i < argc; i++) {
        Buffer<uint8_t> image = load_image(std::string(argv[1]) + "/" + std::string(argv[i]));
        im.push_back(image);

    }
    std::cout << "input image size = " << im[0].width() << "," << im[0].height() << std::endl;

    int lenim = argc - 2;
    //int mid = (lenim-1) / 2;
    int mid = lenim / 2;
    int max_diam = 4;
    int radius_descriptor = 10;
    std::cout << "number of images = "  << lenim << ", mid = " << mid << std::endl;

    std::vector<int> cb_w;
    std::vector<int> cb_h;
    std::vector<int> cb_tile_x;
    std::vector<int> cb_tile_y;
    std::vector<Func> cb_features;
    std::vector<Func> cb_corners;
    std::vector<Func> cb_inputs;

    // start generator
    // 1. compute harris corners, and features around corners
    for (int i = 0; i < lenim; i++) {
        
        int w = im[i].width();
        int h = im[i].height();
        int tile_x = w / max_diam;
        int tile_y = h / max_diam;

        Func input_redge = BoundaryConditions::repeat_edge(im[i], {{0, w}, {0, h}});
        Func lum = calculate_luminance(input_redge, "lum");
        Func corners = harris_corners(lum, w, h, max_diam, 0.15, 1, 5, 5);
        Func features = compute_feature(lum, w, h, corners, 3.f, radius_descriptor);

        cb_w.push_back(w);
        cb_h.push_back(h);
        cb_tile_x.push_back(tile_x);
        cb_tile_y.push_back(tile_y);
        cb_corners.push_back(corners);
        cb_features.push_back(features);
        cb_inputs.push_back(input_redge);

        /*{
            // Visualize harris corner
            Func viz_corner = visualize_harris(input_redge, corners, tile_x, tile_y, w, h);
            Buffer<uint8_t> visual_corner1 = viz_corner.realize({w, h, 3});
            save_image(visual_corner1, "im"+ std::to_string(i) +"_harris.png");
        }*/
    }

    // 2. compute correspondence features and use ransac find homography matix
    std::vector<Func> cb_H(lenim);
    std::vector<Func> cb_invH(lenim);
    std::vector<Func> cb_pair(lenim);
    std::vector<Func> cb_bbox(lenim);
    {
        cb_H[mid] = onesH();
        Func box_mid;
        Var c;
        box_mid(c) = 0;
        box_mid(1) = cb_w[mid]-1;
        box_mid(3) = cb_h[mid]-1;
        cb_bbox[mid] = box_mid;
    }
    if (mid < lenim-1) {
        std::cout<< "check point 1-1" << std::endl;
        cb_pair[mid+1] = find_correspondences(cb_features[mid], cb_features[mid+1],
                                              cb_corners[mid], cb_corners[mid+1], 
                                              max_diam, cb_tile_x[mid], cb_tile_y[mid], radius_descriptor, 2.0);
        cb_H[mid+1] = ransac(cb_pair[mid+1], cb_tile_x[mid+1], cb_tile_y[mid+1], 5000, 4.f);
        cb_invH[mid+1] = inverse_homography(cb_H[mid+1]);
        cb_bbox[mid+1] = compute_bbox(cb_bbox[mid], cb_invH[mid+1]);
    }
    if (mid > 0) {
        std::cout<< "check point 1-2" << std::endl;
        cb_pair[mid-1] = find_correspondences(cb_features[mid], cb_features[mid-1],
                                              cb_corners[mid], cb_corners[mid-1], 
                                              max_diam, cb_tile_x[mid], cb_tile_y[mid], radius_descriptor, 2.0);
        cb_H[mid-1] = ransac(cb_pair[mid-1], cb_tile_x[mid-1], cb_tile_y[mid-1], 5000, 4.f);
        cb_invH[mid-1] = inverse_homography(cb_H[mid-1]);
        cb_bbox[mid-1] = compute_bbox(cb_bbox[mid], cb_invH[mid-1]);
    }

    for (int i = 0; i < lenim; i++) {
        int m, n;
        if (i < mid - 1) {
            m = mid - 1 - i;
            n = mid - 2 - i;;
        } else if (i > mid + 1) {
            m = i - 1;
            n = i;
        } else {
            continue;
        }
        std::cout << "i=" << i << ", m=" << m << ", n=" << n << std::endl;
        cb_pair[n] = find_correspondences(cb_features[m], cb_features[n], 
                                          cb_corners[m], cb_corners[n], 
                                          max_diam, cb_tile_x[m], cb_tile_y[m], radius_descriptor, 2.0);
        Func trans_pair = translate_correspondence(cb_pair[n], cb_invH[m]);
        Func H = ransac(trans_pair, cb_tile_x[n], cb_tile_y[n], 5000, 4.f);

        cb_H[n] = H;
        cb_invH[n] = inverse_homography(cb_H[n]);
        cb_bbox[n] = compute_bbox(cb_bbox[mid], cb_invH[n]);
    }

    // 3. Find output boundary box
    Func u_box = bbox_union(cb_bbox);
    Buffer<int> out_ubox = u_box.realize({4});
    std::cout << "outputbox: " << out_ubox(0) << ", " << out_ubox(1) << ", " << out_ubox(2) << ", "<< out_ubox(3) << std::endl;
    
    int output_width =  (out_ubox(1) - out_ubox(0) +1) < 10000 ?  (out_ubox(1) - out_ubox(0) + 1) : 10000;
    int output_height = (out_ubox(3) - out_ubox(2) +1) < 10000 ?  (out_ubox(3) - out_ubox(2) + 1) : 10000;
    
    // 4. Transform images to output bbox coordinate
    std::vector<Func> cb_transweight(lenim);
    std::vector<Func> cb_transim(lenim);
    std::vector<Func> cb_lowfrequency(lenim);
    std::vector<Func> cb_highfrequency(lenim);

    Func translate("translate_matrix");
    Var i, j;
    translate(i, j) = 0.f;
    translate(0, 0) = 1.f; translate(2, 0) = f32(u_box(0)); 
    translate(1, 1) = 1.f; translate(2, 1) = f32(u_box(2));
    translate(2, 2) = 1.f;

    Func inv_translate("translate_matrix");
    inv_translate(i, j) = 0.f;
    inv_translate(0, 0) = 1.f; inv_translate(2, 0) = f32(-u_box(0)); 
    inv_translate(1, 1) = 1.f; inv_translate(2, 1) = f32(-u_box(2));
    inv_translate(2, 2) = 1.f;
    
    for (int i = 0; i < lenim; i++) {
        Func final_H = prodH(cb_H[i], translate);
        Func weight = calculate_weight(cb_w[i], cb_h[i]);
        cb_transweight[i] = apply_finalH(weight, final_H, cb_w[i], cb_h[i]);
        
        if (i == 0) {

            cb_transim[i] = apply_finalH(cb_inputs[i], final_H, cb_w[i], cb_h[i]);
            cb_lowfrequency[i] = apply_finalH(blur_gaussian(cb_inputs[i], 15, "low_frequency"+std::to_string(i)),
                                              final_H, cb_w[i], cb_h[i]);
        } else {

            Func transim_temp = apply_finalH(cb_inputs[i], final_H, cb_w[i], cb_h[i]);
            cb_transim[i] = color_correction(cb_transim[i-1], transim_temp, 
                                             cb_transweight[i-1], cb_transweight[i], 
                                             output_width, output_height);
            
            Func lowfreqeuncy_temp = apply_finalH(blur_gaussian(cb_inputs[i], 15, "low_frequency"+std::to_string(i)),
                                                  final_H, cb_w[i], cb_h[i]);
            cb_lowfrequency[i] = color_correction(cb_lowfrequency[i-1], lowfreqeuncy_temp, 
                                                  cb_transweight[i-1], cb_transweight[i], 
                                                  output_width, output_height);
        }

        cb_highfrequency[i] = calculate_highfrequency(cb_transim[i], cb_lowfrequency[i]);

        /*{
            Func out_img = to_uint8(cb_lowfrequency[i]);
            Buffer<uint8_t> outimage = out_img.realize({output_width, output_height, 3});
            save_image(outimage, "correction_low" + std::to_string(i) + ".png");
        }*/
    }

    // 5. Stich images
    std::vector<Func> cb_stitchhigh(lenim);
    std::vector<Func> cb_stitchlow(lenim);
    std::vector<Func> cb_stichweight(lenim);

    cb_stitchhigh[0] = cb_highfrequency[0];
    cb_stitchlow[0] = cb_lowfrequency[0];
    cb_stichweight[0] = cb_transweight[0];
    
    for (int i = 1; i < lenim; i++) {
        cb_stitchhigh[i] = stitch_high( cb_stitchhigh[i-1], cb_highfrequency[i],
                                       cb_stichweight[i-1],  cb_transweight[i]);
        cb_stitchlow[i] = stitch_low( cb_stitchlow[i-1], cb_lowfrequency[i],
                                    cb_stichweight[i-1],  cb_transweight[i]);
        cb_stichweight[i] = combine_weight(cb_stichweight[i-1],  cb_transweight[i]);
    }
    Func final_image = combine_weight(cb_stitchhigh[lenim-1], cb_stitchlow[lenim-1]);

    Func out_img = to_uint8(final_image);
    Buffer<uint8_t> outimage = out_img.realize({output_width, output_height, 3});
    
    std::string image_name = std::string(argv[2]);
    std::string output_name = image_name.substr(0, image_name.length() - 5);

    save_image(outimage, "output_images/" + output_name + "panorama.png");
    
    printf("Success!!\n");
    return 0;
}