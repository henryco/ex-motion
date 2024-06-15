//
// Created by henryco on 6/4/24.
//

#ifndef XMOTION_OCL_KERNELS_H
#define XMOTION_OCL_KERNELS_H

#include <string>

namespace xm::ocl::kernels {

    inline const std::string GAUSSIAN_BLUR_KERNEL = R"C(
__kernel void gaussian_blur_horizontal(
        __global const unsigned char *input,
        __global const float *gaussian_kernel,
        __global unsigned char *output,
        const unsigned int width,
        const unsigned int height,
        const int half_kernel_size
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    float sum[3] = {0.f, 0.f, 0.f};
    float weight_sum = 0.f;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        const float weight = gaussian_kernel[k + half_kernel_size];
        const int ix = clamp((int) x + k, (int) 0, (int) width - 1);
        const int idx = (y * width + ix) * 3; // [(B G R),(B G R),(B G R)...]

        sum[0] += input[idx + 0] * weight;
        sum[1] += input[idx + 1] * weight;
        sum[2] += input[idx + 2] * weight;
        weight_sum += weight;
    }

    const int idx = (y * width + x) * 3;
    output[idx + 0] = (unsigned char) (sum[0] / weight_sum);
    output[idx + 1] = (unsigned char) (sum[1] / weight_sum);
    output[idx + 2] = (unsigned char) (sum[2] / weight_sum);
}

__kernel void gaussian_blur_vertical(
        __global const unsigned char *input,
        __global const float *gaussian_kernel,
        __global unsigned char *output,
        const unsigned int width,
        const unsigned int height,
        const int half_kernel_size
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    float sum[3] = {0.f, 0.f, 0.f};
    float weight_sum = 0.f;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        const float weight = gaussian_kernel[k + half_kernel_size];
        const int iy = clamp((int) y + k, (int) 0, (int) height - 1);
        const int idx = (iy * width + x) * 3; // [(B G R),(B G R),(B G R)...]

        sum[0] += input[idx + 0] * weight;
        sum[1] += input[idx + 1] * weight;
        sum[2] += input[idx + 2] * weight;
        weight_sum += weight;
    }

    const int idx = (y * width + x) * 3;
    output[idx + 0] = (unsigned char) (sum[0] / weight_sum);
    output[idx + 1] = (unsigned char) (sum[1] / weight_sum);
    output[idx + 2] = (unsigned char) (sum[2] / weight_sum);
}
)C";

    inline const std::string DILATE_GRAY_KERNEL = R"C(
__kernel void dilate_horizontal(
        __global const unsigned char *input,
        __global unsigned char *output,
        const int half_kernel_size,
        const unsigned int width,
        const unsigned int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    unsigned char max_val = 0;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        const int ix = x + k;
        if (ix < 0 || ix >= width)
            continue;
        max_val = max(max_val, input[y * width + ix]);
    }
    output[y * width + x] = max_val;
}

__kernel void dilate_vertical(
        __global const unsigned char *input,
        __global unsigned char *output,
        const int half_kernel_size,
        const unsigned int width,
        const unsigned int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    unsigned char max_val = 0;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        const int iy = y + k;
        if (iy < 0 || iy >= height)
            continue;
        max_val = max(max_val, input[iy * width + x]);
    }
    output[y * width + x] = max_val;
}
)C";

    inline const std::string ERODE_GRAY_KERNEL = R"C(
__kernel void erode_horizontal(
        __global const unsigned char *input,
        __global unsigned char *output,
        const int half_kernel_size,
        const unsigned int width,
        const unsigned int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    unsigned char min_val = 255;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        const int ix = x + k;
        if (ix < 0 || ix >= width)
            continue;
        min_val = min(min_val, input[y * width + ix]);
    }
    output[y * width + x] = min_val;
}

__kernel void erode_vertical(
        __global const unsigned char *input,
        __global unsigned char *output,
        const int half_kernel_size,
        const unsigned int width,
        const unsigned int height
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    unsigned char min_val = 255;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        const int iy = y + k;
        if (iy < 0 || iy >= height)
            continue;
        min_val = min(min_val, input[iy * width + x]);
    }
    output[y * width + x] = min_val;
}
)C";

    inline const std::string MASK_APPLY_BG_FG = R"C(
__kernel void apply_mask(
        __global const unsigned char *mask_gray,
        __global const unsigned char *front_bgr,
        __global unsigned char *output_bgr,
        const unsigned int mask_width,
        const unsigned int mask_height,
        const unsigned int width,
        const unsigned int height,
        const float scale_mask_w,
        const float scale_mask_h,
        const unsigned char color_b,
        const unsigned char color_g,
        const unsigned char color_r
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int m_x = x * scale_mask_w;
    const int m_y = y * scale_mask_h;

    const int idx = y * width + x;
    const int mps = m_y * mask_width + m_x;
    const int pix = idx * 3;

    if (mask_gray[mps] > 0) {
        output_bgr[pix + 0] = color_b;
        output_bgr[pix + 1] = color_g;
        output_bgr[pix + 2] = color_r;
    } else {
        output_bgr[pix + 0] = front_bgr[pix + 0];
        output_bgr[pix + 1] = front_bgr[pix + 1];
        output_bgr[pix + 2] = front_bgr[pix + 2];
    }
}
)C";

}

#endif //XMOTION_OCL_KERNELS_H
