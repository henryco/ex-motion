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
        eox::ocl::Kernel ocl_dilate_gray;
        eox::ocl::Kernel ocl_erode_gray;
        eox::ocl::Kernel ocl_mask_color;

        /* ==================== OPENCL KERNELS ==================== */
        cv::ocl::Kernel gaussian_blur_h;
        cv::ocl::Kernel gaussian_blur_v;

        cv::ocl::Kernel in_range_hls;

        cv::ocl::Kernel dilate_gray_h;
        cv::ocl::Kernel dilate_gray_v;

        cv::ocl::Kernel erode_gray_h;
        cv::ocl::Kernel erode_gray_v;

        cv::ocl::Kernel mask_color;

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

    /**
     * Computes optimal work group size for given problem dimension.
     *
     * \code
size_t g_size[2] = {
     optimal_work_group_size(img_w, 32),
     optimal_work_group_size(img_h, 32)
};
size_t l_size[2] = {32, 32};
kernel.run(2, g_size, l_size, true)
     * \endcode
     *
     * @param src global dimension size, ie. image dimension - width or height
     * @param pref_work_group_size preferred work group size for given GPU (often 32 or 64)
     * @return optimal global size
     */
    size_t optimal_work_group_size(int src, size_t pref_work_group_size);

    /**
     * Gaussian blur with separate horizontal and vertical pass
     * @param in input image in BGR color space (3 channels uchar)
     * @param out output image in BGR color space (3 channels uchar)
     * @param kernel_size should be odd: 3, 5, 7, 9 ... etc
     * @param sigma if 0 calculated from kernel
     */
    void blur(const cv::UMat &in, cv::UMat &out, int kernel_size = 5, float sigma = 0.f);

    /**
     * Returns mask that satisfies HLS range. This function supports HUE wrapping (!)
     * @param hls_low lowest value for pixel in HLS color space uchar
     * @param hls_up  highest value for pixel in HLS color space uchar
     * @param in input image in BGR color space (3 channels uchar)
     * @param out output mask grayscale (1 channel uchar)
     */
    void bgr_in_range_hls(const cv::Scalar &hls_low, const cv::Scalar &hls_up, const cv::UMat &in, cv::UMat &out);

    /**
     * Dilation filter (conv)
     * @param in input image in grayscale (single channel uchar)
     * @param out output image in grayscale (single channel uchar)
     * @param iterations numbers of iterations to apply this filter
     * @param kernel_size kernel size for filter
     */
    void dilate(const cv::UMat &in, cv::UMat &out, int iterations = 1, int kernel_size = 3);

    /**
     * Erosion filter (conv)
     * @param in input image in grayscale (single channel uchar)
     * @param out output image in grayscale (single channel uchar)
     * @param iterations numbers of iterations to apply this filter
     * @param kernel_size kernel size for filter
     */
    void erode(const cv::UMat &in, cv::UMat &out, int iterations = 1, int kernel_size = 3);

    /**
     * Apply mask to image, places for which mask value != 0 would be replaced with given color.
     * @note Useful for chroma-keys
     * @param color color which is used when mask value != 0
     * @param img source image in BGR color space (3 channels uchar)
     * @param mask grayscale mask (single channel uchar), can be smaller or larger than image (!)
     * @param out output image in BGR color space (3 channels uchar)
     */
    void apply_mask_with_color(const cv::Scalar &color, const cv::UMat &img, const cv::UMat &mask, cv::UMat &out);
}

#endif //XMOTION_OCL_FILTERS_H
