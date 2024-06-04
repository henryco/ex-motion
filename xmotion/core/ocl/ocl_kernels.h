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
        const unsigned int kernel_size
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int half_kernel_size = kernel_size / 2;
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
    output[idx + 0] = sum[0] / weight_sum;
    output[idx + 1] = sum[1] / weight_sum;
    output[idx + 2] = sum[2] / weight_sum;
}

__kernel void gaussian_blur_vertical(
        __global const unsigned char *input,
        __global const float *gaussian_kernel,
        __global unsigned char *output,
        const unsigned int width,
        const unsigned int height,
        const unsigned int kernel_size
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int half_kernel_size = kernel_size / 2;
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
    output[idx + 0] = sum[0] / weight_sum;
    output[idx + 1] = sum[1] / weight_sum;
    output[idx + 2] = sum[2] / weight_sum;
}
)C";

    inline const std::string BGR_HLS_RANGE_KERNEL = R"C(
inline void bgr_to_hls(
        const unsigned char *in_bgr,
        unsigned char *out_hls
) {
    const float b = ((float) in_bgr[0]) / 255.f;
    const float g = ((float) in_bgr[1]) / 255.f;
    const float r = ((float) in_bgr[2]) / 255.f;

    const float c_max = max(b, max(g, r));
    const float c_min = min(b, min(g, r));
    const float c_dif = c_max - c_min;
    const float c_sum = c_max + c_min;

    // L
    out_hls[1] = (unsigned char) ((c_sum / 2.f) * 255.f);

    // S
    out_hls[2] = out_hls[1] <= 127
            ? (unsigned char) ((c_dif / c_sum) * 255.f)
            : (unsigned char) ((c_dif / (2.f - c_dif)) * 255.f);

    // H
    if (c_max == c_min) {
        out_hls[0] = 0;
    }
    else if (c_max == b) {
        // 42.5 = 60 / 360 * 255
        out_hls[0] = (unsigned char) ((((r - g) / c_dif) + 4.f) * 42.5f);
    }
    else if (c_max == g) {
        // 42.5 = 60 / 360 * 255
        out_hls[0] = (unsigned char) ((((b - r) / c_dif) + 2.f) * 42.5f);
    }
    else if (c_max == r) {
        // 0.7083333 = 255 / 360
        out_hls[0] = (unsigned char) ((float) (((short) (60.f * ((g - b) / c_dif))) % 360) * 0.7083333f);
    }
}

__kernel void in_range_hls (
        __global const unsigned char *input_bgr,
        __global const unsigned char *lower_hls,
        __global const unsigned char *upper_hls,
        __global unsigned char *output_gray,
        const unsigned int width,
        const unsigned int height,
        const unsigned int is_spin // TODO
) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height)
        return;

    const int idx = y * width + x;

    unsigned char hls[3];
    bgr_to_hls(input_bgr[idx * 3], hls);

    if (hls[0] >= lower_hls[0] &&
        hls[1] >= lower_hls[1] &&
        hls[2] >= lower_hls[2] &&
        hls[0] <= upper_hls[0] &&
        hls[1] <= upper_hls[1] &&
        hls[2] <= upper_hls[2]) {
        output_gray[idx] = 255;
    } else {
        output_gray[idx] = 0;
    }
}

)C";

}

#endif //XMOTION_OCL_KERNELS_H
