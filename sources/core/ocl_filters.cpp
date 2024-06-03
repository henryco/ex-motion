//
// Created by henryco on 6/3/24.
//

#include "../../xmotion/core/ocl/ocl_filters.h"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace xm::ocl {

    Kernels::Kernels() {
        ocl_gaussian_blur.compile(GAUSSIAN_BLUR_KERNEL);
        ocl_gaussian_blur.procedure("gaussian_blur_horizontal");
        ocl_gaussian_blur.procedure("gaussian_blur_vertical");
    }

    void create_gaussian_kernel(float *kernel, int kernel_size, float sigma) {
        const int half_kernel_size = kernel_size / 2;
        float sum = 0.0f;
        for (int y = -half_kernel_size; y <= half_kernel_size; y++) {
            for (int x = -half_kernel_size; x <= half_kernel_size; x++) {
                const int index = (y + half_kernel_size) * kernel_size + (x + half_kernel_size);
                kernel[index] = std::exp((float) -(x * x + y * y) / (2 * sigma * sigma));
                sum += kernel[index];
            }
        }
        for (int i = 0; i < kernel_size * kernel_size; i++) {
            kernel[i] /= sum;
        }
    }

    void blur(const cv::UMat &in, cv::UMat &out, const int kernel_size, float sigma) {
        if (sigma == 0)
            sigma = ((float) kernel_size - 1.f) / 6.f;

        cv::UMat result(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        cv::UMat kernel_mat;
        cv::getGaussianKernel(kernel_size, sigma, CV_32F).copyTo(kernel_mat);

        const size_t pref_size = xm::ocl::Kernels::getInstance().ocl_gaussian_blur.get_pref_size();

        auto kernel_h = xm::ocl::Kernels::getInstance()
                .ocl_gaussian_blur.get_kernel("gaussian_blur_horizontal");
        {
            int idx = 0;
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrReadOnly(kernel_mat));
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result));
            idx = kernel_h.set(idx, (uint) in.cols);
            idx = kernel_h.set(idx, (uint) in.rows);
            kernel_h.set(idx, (uint) kernel_size);

            size_t g_size[2] = {optimal_work_group_size(in.cols, pref_size),
                                optimal_work_group_size(in.rows, pref_size)};
            size_t l_size[2] = {pref_size, pref_size};

            const bool success = kernel_h.run(2, g_size, l_size, true);
            if (!success)
                throw std::runtime_error("opencl kernel error");
        }

        auto kernel_v = xm::ocl::Kernels::getInstance().ocl_gaussian_blur
                .get_kernel("gaussian_blur_vertical");
        {
            int idx = 0;
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(kernel_mat));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result));
            idx = kernel_v.set(idx, (uint) in.cols);
            idx = kernel_v.set(idx, (uint) in.rows);
            kernel_v.set(idx, (uint) kernel_size);

            size_t g_size[2] = {optimal_work_group_size(in.cols, pref_size),
                                optimal_work_group_size(in.rows, pref_size)};
            size_t l_size[2] = {pref_size, pref_size};
            const bool success = kernel_h.run(2, g_size, l_size, true);
            if (!success)
                throw std::runtime_error("opencl kernel error");
        }

        out = result;
    }

    size_t optimal_work_group_size(int src, size_t size) {
        if (src % size == 0)
            return src;
        return src + size - (src % size);
    }

}