//
// Created by henryco on 09/06/24.
//

#ifndef XMOTION_OCL_KERNELS_CHROMAKEY_H
#define XMOTION_OCL_KERNELS_CHROMAKEY_H

#include <string>

namespace xm::ocl::kernels {

    inline const std::string CHROMA_KEY_COMPLEX = R"cl(
inline void downscale(
        const unsigned char *in_img_bgr, // from
        unsigned char *out_pix_bgr,      // to

        const unsigned int img_w,        // from
        const unsigned int img_h,        // from
        const float scale_w,             // [from : to] ie: [2 : 1]
        const float scale_h,             // [from : to] ie: [2 : 1]
        const int pix_x,                 // to
        const int pix_y,                 // to

        const bool linear                // linear or nearest
) {
    if (!linear) {
        const int img_x = clamp((int) (pix_x * scale_w), (int) 0, (int) (img_w - 1));
        const int img_y = clamp((int) (pix_y * scale_h), (int) 0, (int) (img_h - 1));
        const int pos = ((img_y * img_w) + img_x) * 3;
        out_pix_bgr[0] = in_img_bgr[pos + 0];
        out_pix_bgr[1] = in_img_bgr[pos + 1];
        out_pix_bgr[2] = in_img_bgr[pos + 2];
        return;
    }

    const float img_x = pix_x * scale_w;
    const float img_y = pix_y * scale_h;
    const int pos = ((img_y * img_w) + img_x) * 3;

    const int x0 = (int) img_x;
    const int y0 = (int) img_y;
    const float dx = img_x - x0;
    const float dy = img_y - y0;

    const int x1 = min((int) (x0 + 1), (int) (img_w - 1));
    const int y1 = min((int) (y0 + 1), (int) (img_h - 1));

    for (int i = 0; i < 3; i++) {
        float p00 = in_img_bgr[(y0 * img_w + x0) * 3 + i];
        float p10 = in_img_bgr[(y0 * img_w + x1) * 3 + i];
        float p01 = in_img_bgr[(y1 * img_w + x0) * 3 + i];
        float p11 = in_img_bgr[(y1 * img_w + x1) * 3 + i];

        float p0 = p00 + (p10 - p00) * dx;
        float p1 = p01 + (p11 - p01) * dx;
        float p = p0 + (p1 - p0) * dy;
        out_pix_bgr[i] = (unsigned char) p;
    }
}

inline void bgr_to_hls(
        const unsigned char *in_bgr,
        unsigned char *out_hls
) {
    // https://docs.opencv.org/4.9.0/de/d25/imgproc_color_conversions.html
    const float b = ((float) in_bgr[0]) / 255.f;
    const float g = ((float) in_bgr[1]) / 255.f;
    const float r = ((float) in_bgr[2]) / 255.f;

    const float c_max = fmax(b, fmax(g, r));
    const float c_min = fmin(b, fmin(g, r));
    const float c_dif = c_max - c_min;
    const float c_sum = c_max + c_min;
    const float c_fac = 60.f / c_dif;
    const float L = (c_sum / 2.f);

    float H = 0.f;
    if (c_max == r)
        H = c_fac * (g - b) ;
    else if (c_max == g)
        H = 120.f + (c_fac * (b - r));
    else if (c_max == b)
        H = 240.f + (c_fac * (r - g));
    if (H < 0)
        H += 360.f;

    out_hls[0] = (unsigned char) (H * 0.708333f); // 255 / 360
    out_hls[1] = (unsigned char) (L * 255.f);
    out_hls[2] = (unsigned char) ((L < 0.5f ? (c_dif / c_sum) : (c_dif / (2.f - c_sum))) * 255.f);
}

inline unsigned char scale_blur_hls_threshold(
        const unsigned char *input,
        const float *gaussian_kernel,
        const int half_kernel_size,
        const bool blur,
        const bool linear,
        const unsigned int input_w,
        const unsigned int input_h,
        const unsigned int mask_w,
        const unsigned int mask_h,
        const float scale_w,
        const float scale_h,
        const unsigned char lower_h,
        const unsigned char lower_l,
        const unsigned char lower_s,
        const unsigned char upper_h,
        const unsigned char upper_l,
        const unsigned char upper_s,
        const int x,
        const int y
) {
    unsigned char pix_bgr[3] = {0, 0, 0}; // working pixel

    if (blur) { // ================================== BLUR ========================================
        float sum[3] = {0.f, 0.f, 0.f};
        float weight_sum = 0.f;

        for (int ky = -half_kernel_size; ky <= half_kernel_size; ky++) {
            for (int kx = -half_kernel_size; kx <= half_kernel_size; kx++) {
                const int i_x = x + kx;
                const int i_y = y + ky;

                if (i_x >= 0 && i_x < mask_w && i_y >= 0 && i_y < mask_h) {
                    const float weight = gaussian_kernel[half_kernel_size + kx] * gaussian_kernel[half_kernel_size + ky];
                    unsigned char pixel[3] = {0, 0, 0};

                    downscale(input, pixel, input_w, input_h, scale_w, scale_h, i_x, i_y, linear);

                    sum[0] += ((float) pixel[0]) * weight;
                    sum[1] += ((float) pixel[1]) * weight;
                    sum[2] += ((float) pixel[2]) * weight;
                    weight_sum += weight;
                }
            }
        }

        pix_bgr[0] = (unsigned char) (sum[0] / weight_sum);
        pix_bgr[1] = (unsigned char) (sum[1] / weight_sum);
        pix_bgr[2] = (unsigned char) (sum[2] / weight_sum);

    } else { // ================================== NON-BLUR ========================================
        downscale(input, pix_bgr, input_w, input_h, scale_w, scale_h, x, y, linear);
    }

    // ================================== HLS MASK ========================================

    unsigned char hls[3];
    bgr_to_hls(pix_bgr, hls);

    unsigned char mask = 0;
    if (lower_h < upper_h) { // Dont wrap
        if (hls[0] >= lower_h &&
            hls[1] >= lower_l &&
            hls[2] >= lower_s &&
            hls[0] <= upper_h &&
            hls[1] <= upper_l &&
            hls[2] <= upper_s) {
            return 255;
        } else {
            return 0;
        }
    }

    else { // WRAP !
        if ((hls[0] >= lower_h || hls[0] <= upper_h) &&
            hls[1] >= lower_l &&
            hls[2] >= lower_s &&
            hls[1] <= upper_l &&
            hls[2] <= upper_s) {
            return 255;
        } else {
            return 0;
        }
    }
}

__kernel void power_mask(
        // IMAGES
        __global const unsigned char *input,
        __global unsigned char *output,

        // BLUR
        __global const float *gaussian_kernel,
        const int half_kernel_size,
        const unsigned char blur,   // aka BOOL

        // SCALING
        const unsigned char linear, // aka BOOL
        const int input_w,
        const int input_h,
        const int mask_w,
        const int mask_h,
        const float scale_w,
        const float scale_h,

        // HLS MASK
        const unsigned char lower_h,
        const unsigned char lower_l,
        const unsigned char lower_s,
        const unsigned char upper_h,
        const unsigned char upper_l,
        const unsigned char upper_s
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= mask_w || y >= mask_h)
        return;

    output[y * mask_w + x] = scale_blur_hls_threshold(
            input,
            gaussian_kernel,
            half_kernel_size,
            blur > 0,
            linear > 0,
            input_w, input_h,
            mask_w, mask_h,
            scale_w, scale_h,
            lower_h, lower_l, lower_s,
            upper_h, upper_l, upper_s,
            x, y);
}

__kernel void power_chromakey(
        // IMAGES
        __global const unsigned char *input,
        __global unsigned char *output,

        // BLUR
        __global const float *gaussian_kernel,
        const int half_kernel_size,
        const unsigned char blur,   // aka BOOL

        // SCALING
        const unsigned char linear, // aka BOOL
        const int input_w,
        const int input_h,
        const int mask_w,
        const int mask_h,
        const float scale_x,
        const float scale_y,

        // HLS MASK
        const unsigned char lower_h,
        const unsigned char lower_l,
        const unsigned char lower_s,
        const unsigned char upper_h,
        const unsigned char upper_l,
        const unsigned char upper_s,

        // KEY
        const unsigned char color_b,
        const unsigned char color_g,
        const unsigned char color_r,

        // UPSCALE
        const unsigned int d_x,
        const unsigned int d_y
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= mask_w || y >= mask_h)
        return;

    const unsigned char mask = scale_blur_hls_threshold(
            input,
            gaussian_kernel,
            half_kernel_size,
            blur > 0,
            linear > 0,
            input_w, input_h,
            mask_w, mask_h,
            scale_x, scale_y,
            lower_h, lower_l, lower_s,
            upper_h, upper_l, upper_s,
            x, y);

    const float ox = x * scale_x;
    const float oy = y * scale_y;

    for (int ky = 0; ky < d_y; ky++) {
        for (int kx = 0; kx < d_x; kx++) {
            const int ix = clamp((int) (ox + kx), (int) 0, (int) (input_w - 1));
            const int iy = clamp((int) (oy + ky), (int) 0, (int) (input_h - 1));
            const int idx = (iy * input_w + ix) * 3;
            if (mask > 0) {
                output[idx + 0] = color_b;
                output[idx + 1] = color_g;
                output[idx + 2] = color_r;
            } else {
                output[idx + 0] = input[idx + 0];
                output[idx + 1] = input[idx + 1];
                output[idx + 2] = input[idx + 2];
            }
        }
    }
})cl";

}

#endif //XMOTION_OCL_KERNELS_CHROMAKEY_H
