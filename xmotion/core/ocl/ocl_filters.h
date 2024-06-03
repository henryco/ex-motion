//
// Created by henryco on 6/3/24.
//

#ifndef XMOTION_OCL_FILTERS_H
#define XMOTION_OCL_FILTERS_H

#include <opencv2/core/mat.hpp>
#include <string>
#include "kernel.h"

namespace xm::ocl {

    inline const std::string GAUSSIAN_BLUR_KERNEL = R"ocl(
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
)ocl";

    class Kernels {
    public:
        /* =================== KERNELS WRAPPERS =================== */
        eox::ocl::Kernel ocl_gaussian_blur;

        /* ==================== OPENCL KERNELS ==================== */
        cv::ocl::Kernel gaussian_blur_h;
        cv::ocl::Kernel gaussian_blur_v;

        static Kernels& getInstance() {
            static Kernels instance;
            return instance;
        }

        Kernels(const Kernels &) = delete;
        Kernels(const Kernels &&) = delete;
        Kernels &operator=(const Kernels &) = delete;
        [[nodiscard]] size_t get_pref_work_group_size() const;

    private:
        size_t pref_work_group_size = 0;
        Kernels();
    };

    size_t optimal_work_group_size(int src, size_t size);

    void blur(const cv::UMat &in, cv::UMat &out, int kernel_size = 5, float sigma = 0.f);
}

#endif //XMOTION_OCL_FILTERS_H
