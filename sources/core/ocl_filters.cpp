//
// Created by henryco on 6/3/24.
//

#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/ocl/ocl_kernels.h"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace xm::ocl {

    Kernels::Kernels() {
        ocl_gaussian_blur.compile(xm::ocl::kernels::GAUSSIAN_BLUR_KERNEL);
        gaussian_blur_h = ocl_gaussian_blur.procedure("gaussian_blur_horizontal");
        gaussian_blur_v = ocl_gaussian_blur.procedure("gaussian_blur_vertical");

        ocl_in_range_hls.compile(xm::ocl::kernels::BGR_HLS_RANGE_KERNEL);
        in_range_hls = ocl_in_range_hls.procedure("in_range_hls");

        ocl_dilate_gray.compile(xm::ocl::kernels::DILATE_GRAY_KERNEL);
        dilate_gray_h = ocl_dilate_gray.procedure("dilate_horizontal");
        dilate_gray_v = ocl_dilate_gray.procedure("dilate_vertical");

        ocl_erode_gray.compile(xm::ocl::kernels::ERODE_GRAY_KERNEL);
        erode_gray_h = ocl_erode_gray.procedure("erode_horizontal");
        erode_gray_v = ocl_erode_gray.procedure("erode_vertical");

        ocl_mask_color.compile(xm::ocl::kernels::MASK_APPLY_BG_FG);
        mask_color = ocl_mask_color.procedure("apply_mask");

        pref_work_group_size = ocl_gaussian_blur.get_pref_size();
    }

    size_t Kernels::get_pref_work_group_size() const {
        return pref_work_group_size;
    }

    size_t optimal_work_group_size(int src, size_t size) {
        if (src % size == 0)
            return src;
        return src + size - (src % size);
    }

    void run_kernel(cv::ocl::Kernel &kernel, int w, int h) {
        const size_t pref_size = xm::ocl::Kernels::getInstance().get_pref_work_group_size();
        size_t g_size[2] = {optimal_work_group_size(w, pref_size), optimal_work_group_size(h, pref_size)};
        size_t l_size[2] = {pref_size, pref_size};
        if (!kernel.run(2, g_size, l_size, true))
            throw std::runtime_error("opencl kernel error");
    }

    void blur(const cv::UMat &in, cv::UMat &out, const int kernel_size, float sigma) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        if (sigma <= 0)
            sigma = ((float) kernel_size - 1.f) / 6.f;

        cv::UMat result_1(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

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
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result_1));
            idx = kernel_h.set(idx, (uint) in.cols);
            idx = kernel_h.set(idx, (uint) in.rows);
            kernel_h.set(idx, (uint) kernel_size);
            if (!kernel_h.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");
        }

        auto kernel_v = xm::ocl::Kernels::getInstance().gaussian_blur_v;
        {
            int idx = 0;
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(result_1));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(kernel_mat));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result_2));
            idx = kernel_v.set(idx, (uint) in.cols);
            idx = kernel_v.set(idx, (uint) in.rows);
            kernel_v.set(idx, (uint) kernel_size);
            if (!kernel_v.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");
        }

        out = result_2;
    }

    void bgr_in_range_hls(const cv::Scalar &hls_low, const cv::Scalar &hls_up, const cv::UMat &in, cv::UMat &out) {
        cv::UMat result(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat bot(1, 1, CV_8UC3, hls_low, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat top(1, 1, CV_8UC3, hls_up, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kernel = xm::ocl::Kernels::getInstance().in_range_hls;
        {
            int idx = 0;
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrReadOnly(bot));
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrReadOnly(top));
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result));
            idx = kernel.set(idx, (uint) in.cols);
            kernel.set(idx, (uint) in.rows);
            run_kernel(kernel, in.cols, in.rows);
        }
        out = result;
    }

    void dilate(const cv::UMat &in, cv::UMat &out, int iterations, int kernel_size) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        cv::UMat result_1(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        const size_t pref_size = xm::ocl::Kernels::getInstance().get_pref_work_group_size();
        size_t g_size[2] = {optimal_work_group_size(in.cols, pref_size),
                            optimal_work_group_size(in.rows, pref_size)};
        size_t l_size[2] = {pref_size, pref_size};

        auto kernel_h = xm::ocl::Kernels::getInstance().dilate_gray_h;
        auto kernel_v = xm::ocl::Kernels::getInstance().dilate_gray_v;

        {
            int idx = 0;
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result_1));
            idx = kernel_h.set(idx, (uint) kernel_size);
            idx = kernel_h.set(idx, (uint) in.cols);
            kernel_h.set(idx, (uint) in.rows);
        }

        {
            int idx = 0;
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(result_1));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result_2));
            idx = kernel_v.set(idx, (uint) kernel_size);
            idx = kernel_v.set(idx, (uint) in.cols);
            kernel_v.set(idx, (uint) in.rows);
        }

        for (int i = 0; i < iterations; i++) {
            if (!kernel_h.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");
            if (!kernel_v.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");

            if (i == 0 && iterations > 1)
                kernel_h.set(0, cv::ocl::KernelArg::PtrReadOnly(result_2));
        }

        out = result_2;
    }

    void erode(const cv::UMat &in, cv::UMat &out, int iterations, int kernel_size) {
        if (kernel_size < 3 || kernel_size % 2 == 0)
            throw std::runtime_error("Invalid kernel size: " + std::to_string(kernel_size));

        cv::UMat result_1(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat result_2(in.rows, in.cols, CV_8UC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        const size_t pref_size = xm::ocl::Kernels::getInstance().get_pref_work_group_size();
        size_t g_size[2] = {optimal_work_group_size(in.cols, pref_size),
                            optimal_work_group_size(in.rows, pref_size)};
        size_t l_size[2] = {pref_size, pref_size};

        auto kernel_h = xm::ocl::Kernels::getInstance().erode_gray_h;
        auto kernel_v = xm::ocl::Kernels::getInstance().erode_gray_v;

        {
            int idx = 0;
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrReadOnly(in));
            idx = kernel_h.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result_1));
            idx = kernel_h.set(idx, (uint) kernel_size);
            idx = kernel_h.set(idx, (uint) in.cols);
            kernel_h.set(idx, (uint) in.rows);
        }

        {
            int idx = 0;
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrReadOnly(result_1));
            idx = kernel_v.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result_2));
            idx = kernel_v.set(idx, (uint) kernel_size);
            idx = kernel_v.set(idx, (uint) in.cols);
            kernel_v.set(idx, (uint) in.rows);
        }

        for (int i = 0; i < iterations; i++) {
            if (!kernel_h.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");
            if (!kernel_v.run(2, g_size, l_size, true))
                throw std::runtime_error("opencl kernel error");

            if (i == 0 && iterations > 1)
                kernel_h.set(0, cv::ocl::KernelArg::PtrReadOnly(result_2));
        }

        out = result_2;
    }

    void apply_mask_with_color(const cv::Scalar &color, const cv::UMat &img, const cv::UMat &mask, cv::UMat &out) {
        cv::UMat result(img.rows, img.cols, CV_8UC3, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
        cv::UMat clr(1, 1, CV_8UC3, color, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

        auto kernel = xm::ocl::Kernels::getInstance().mask_color;
        {
            int idx = 0;
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrReadOnly(mask));
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrReadOnly(img));
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrReadOnly(clr));
            idx = kernel.set(idx, cv::ocl::KernelArg::PtrWriteOnly(result));
            idx = kernel.set(idx, (uint) mask.cols);
            idx = kernel.set(idx, (uint) mask.rows);
            idx = kernel.set(idx, (uint) img.cols);
            idx = kernel.set(idx, (uint) img.rows);
            idx = kernel.set(idx, (float) mask.cols / (float) img.cols);
            kernel.set(idx, (float) mask.rows / (float) img.rows);
            run_kernel(kernel, img.cols, img.rows);
        }

        out = result;
    }

}