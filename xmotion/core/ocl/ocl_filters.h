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
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > width || y > height)
        return;

    int half_kernel_size = kernel_size / 2;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        int ix = clamp((int) x + k, (int) 0, (int) width - 1);
        float weight = gaussian_kernel[k + half_kernel_size];
        sum += input[y * width + ix] * weight;
        weight_sum += weight;
    }

    output[y * width + x] = sum / weight_sum;
}

__kernel void gaussian_blur_vertical(
        __global const unsigned char *input,
        __global const float *gaussian_kernel,
        __global unsigned char *output,
        const unsigned int width,
        const unsigned int height,
        const unsigned int kernel_size
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x > width || y > height)
        return;

    int half_kernel_size = kernel_size / 2;

    float sum = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -half_kernel_size; k <= half_kernel_size; k++) {
        int iy = clamp((int) y + k, (int) 0, (int) height - 1);
        float weight = gaussian_kernel[k + half_kernel_size];
        sum += input[iy * width + x] * weight;
        weight_sum += weight;
    }

    output[y * width + x] = sum / weight_sum;
}
)ocl";

    class Kernels {
    public:
        eox::ocl::Kernel ocl_gaussian_blur;

        static Kernels& getInstance() {
            static Kernels instance;
            return instance;
        }
        Kernels(const Kernels &) = delete;
        Kernels(const Kernels &&) = delete;
        Kernels &operator=(const Kernels &) = delete;

    private:
        Kernels();
    };

    size_t optimal_work_group_size(int src, size_t size);

    void create_gaussian_kernel(float *kernel, int kernel_size, float sigma);

    void blur(const cv::UMat &in, cv::UMat &out, int kernel_size = 5, float sigma = 0.f);
}

#endif //XMOTION_OCL_FILTERS_H
