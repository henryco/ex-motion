//
// Created by henryco on 6/3/24.
//

#include "../../xmotion/core/ocl/ocl_filters.h"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace xm::ocl {

    Kernels::Kernels() {
        ocl_gaussian_blur.compile(GAUSSIAN_BLUR_KERNEL);
        gaussian_blur_h = ocl_gaussian_blur.procedure("gaussian_blur_horizontal");
        gaussian_blur_v = ocl_gaussian_blur.procedure("gaussian_blur_vertical");
        pref_work_group_size = ocl_gaussian_blur.get_pref_size();
    }

    size_t Kernels::get_pref_work_group_size() const {
        return pref_work_group_size;
    }

    void blur(const cv::UMat &in, cv::UMat &out, const int kernel_size, float sigma) {
        if (sigma <= 0)
            sigma = ((float) kernel_size - 1.f) / 6.f;

        cv::UMat result(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        cv::UMat kernel_mat;
        cv::getGaussianKernel(kernel_size, sigma, CV_32F).copyTo(kernel_mat);

        const size_t pref_size = xm::ocl::Kernels::getInstance().get_pref_work_group_size();
        size_t g_size[2] = {optimal_work_group_size(in.cols, pref_size),
                            optimal_work_group_size(in.rows, pref_size)};
        size_t l_size[2] = {pref_size, pref_size};

        auto kernel_h = xm::ocl::Kernels::getInstance().gaussian_blur_h;
        {
            int idx = 0;
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrReadOnly(kernel_mat));
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result));
            idx = kernel_h.set(idx, (uint) in.cols);
            idx = kernel_h.set(idx, (uint) in.rows);
            kernel_h.set(idx, (uint) kernel_size);

            if (!kernel_h.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");
        }

        auto kernel_v = xm::ocl::Kernels::getInstance().gaussian_blur_v;
        {
            int idx = 0;
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(kernel_mat));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result));
            idx = kernel_v.set(idx, (uint) in.cols);
            idx = kernel_v.set(idx, (uint) in.rows);
            kernel_v.set(idx, (uint) kernel_size);

            if (!kernel_h.run(2, g_size, l_size, true))
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