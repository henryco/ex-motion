//
// Created by henryco on 6/3/24.
//

#ifndef XMOTION_OCL_FILTERS_H
#define XMOTION_OCL_FILTERS_H

#include <opencv2/core/mat.hpp>
#include <string>
#include "kernel.h"

namespace xm::ocl {

    class Kernels {
    public:
        /* =================== KERNELS WRAPPERS =================== */
        eox::ocl::Kernel ocl_gaussian_blur;
        eox::ocl::Kernel ocl_in_range_hls;

        /* ==================== OPENCL KERNELS ==================== */
        cv::ocl::Kernel gaussian_blur_h;
        cv::ocl::Kernel gaussian_blur_v;
        cv::ocl::Kernel in_range_hls;

        static Kernels &getInstance() {
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

    void run_kernel(cv::ocl::Kernel &kernel, int w, int h);

    size_t optimal_work_group_size(int src, size_t size);

    /**
     * Gaussian blur with separate horizontal and vertical pass
     * @param in input image in BGR color space (3 channels uchar)
     * @param out output image in BGR color space (3 channels uchar)
     * @param kernel_size should be odd: 3, 5, 7, 9 ... etc
     * @param sigma if 0 calculated from kernel
     */
    void blur(const cv::UMat &in, cv::UMat &out, int kernel_size = 5, float sigma = 0.f);

    /**
     * Returns mask that satisfies HLS range
     * @param hls_low lowest value for pixel in HLS color space uchar
     * @param hls_up  highest value for pixel in HLS color space uchar
     * @param in input image in BGR color space (3 channels uchar)
     * @param out output mask grayscale (1 channel uchar)
     */
    void bgr_in_range_hls(const cv::UMat &hls_low, const cv::UMat &hls_up, const cv::UMat &in, cv::UMat &out);

}

#endif //XMOTION_OCL_FILTERS_H
