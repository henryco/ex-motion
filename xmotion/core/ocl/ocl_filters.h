//
// Created by henryco on 6/3/24.
//

#ifndef XMOTION_OCL_FILTERS_H
#define XMOTION_OCL_FILTERS_H

#include <opencv2/core/mat.hpp>
#include <string>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <CL/cl.h>

#include "kernel.h"
#include "cl_kernel.h"
#include "ocl_data.h"
#include "ocl_interop.h"
#include "../utils/xm_data.h"

namespace xm::ocl {

    namespace aux {
        extern std::mutex GLOBAL_MUTEX;
        inline static bool DEBUG = false;
    }

    class Kernels {
        static inline const auto log =
                spdlog::stdout_color_mt("ocl_filters");
    public:
        std::string thread_id;

        cl_device_id device_id;
        cl_context ocl_context;
        cl_command_queue ocl_command_queue;
        std::map<int, cl_command_queue> ocl_queue_map;

        /* ==================== OPENCL KERNELS ==================== */
        cl_program program_filter_conv;
        cl_kernel kernel_blur_h;
        cl_kernel kernel_blur_v;
        cl_kernel kernel_dilate_h;
        cl_kernel kernel_dilate_v;
        cl_kernel kernel_erode_h;
        cl_kernel kernel_erode_v;
        size_t blur_local_size;
        size_t dilate_local_size;
        size_t erode_local_size;

        cl_program program_color_space;
        cl_kernel kernel_mask_apply;
        cl_kernel kernel_range_hls;
        size_t range_hls_local_size;
        size_t mask_apply_local_size;

        cl_program program_power_chroma;
        cl_kernel kernel_power_chroma;
        cl_kernel kernel_power_apply;
        cl_kernel kernel_power_mask;
        size_t power_chroma_local_size;

        cl_program program_flip_rotate;
        cl_kernel kernel_flip_rotate;
        size_t flip_rotate_local_size;

        cl_program program_background;
        cl_kernel kernel_lbp_texture;
        cl_kernel kernel_lbp_mask_only;
        cl_kernel kernel_lbp_mask_apply;
        cl_kernel kernel_color_diff;
        cl_kernel kernel_lbp_power;
        size_t lbp_local_size;

        /* ==================== CACHE KERNELS ==================== */
        xm::ocl::Image2D blur_kernels[(31 - 1) / 2];

        static Kernels &instance() {
            static thread_local Kernels obj;
            return obj;
        }

        void print_time(cl_ulong time, const std::string &name, bool force = false) const;

        Kernels(const Kernels &) = delete;

        Kernels(const Kernels &&) = delete;

        Kernels &operator=(const Kernels &) = delete;

        ~Kernels();

        cl_command_queue retrieve_queue(int index);

    private:
        Kernels();
    };

    /**
     * Gaussian blur with separate horizontal and vertical pass
     * @param in input image in BGR color space (3 channels uchar)
     * @param kernel_size should be odd: 3, 5, 7, 9 ... etc
     * @param queue_index index of command queue (optional)
     */
    xm::ocl::iop::ClImagePromise blur(
            const xm::ocl::iop::ClImagePromise &in,
            int kernel_size,
            int queue_index = -1);

    /**
     * Gaussian blur with separate horizontal and vertical pass
     * @param queue opencl command queue
     * @param in input image in BGR color space (3 channels uchar)
     * @param kernel_size should be odd: 3, 5, 7, 9 ... etc
     */
    xm::ocl::iop::ClImagePromise blur(
            cl_command_queue queue,
            const xm::ocl::iop::ClImagePromise &in,
            int kernel_size);

    /**
     * Returns mask that satisfies HLS range. This function supports HUE wrapping (!)
     * @param hls_low lowest value for pixel in HLS color space uchar
     * @param hls_up  highest value for pixel in HLS color space uchar
     * @param in input image in BGR color space (3 channels uchar)
     * @param out output mask grayscale (1 channel uchar)
     */
    void bgr_in_range_hls(const cv::Scalar &hls_low,
                          const cv::Scalar &hls_up,
                          const cv::UMat &in,
                          cv::UMat &out,
                          int queue_index = -1);

    /**
     * Dilation filter (conv)
     * @param in input image in grayscale (single channel uchar)
     * @param out output image in grayscale (single channel uchar)
     * @param iterations numbers of iterations to apply this filter
     * @param kernel_size kernel size for filter
     */
    void dilate(const cv::UMat &in,
                cv::UMat &out,
                int iterations = 1,
                int kernel_size = 3,
                int queue_index = -1);

    /**
     * Erosion filter (conv)
     * @param in input image in grayscale (single channel uchar)
     * @param out output image in grayscale (single channel uchar)
     * @param iterations numbers of iterations to apply this filter
     * @param kernel_size kernel size for filter
     */
    void erode(const cv::UMat &in,
               cv::UMat &out,
               int iterations = 1,
               int kernel_size = 3,
               int queue_index = -1);

    /**
     * Apply mask to image, places for which mask value != 0 would be replaced with given color.
     * @note Useful for chroma-keys
     * @param color color which is used when mask value != 0
     * @param img source image in BGR color space (3 channels uchar)
     * @param mask grayscale mask (single channel uchar), can be smaller or larger than image (!)
     * @param out output image in BGR color space (3 channels uchar)
     */
    void apply_mask_with_color(const cv::Scalar &color,
                               const cv::UMat &img,
                               const cv::UMat &mask,
                               cv::UMat &out,
                               int queue_index = -1);

    xm::ocl::iop::ClImagePromise chroma_key(
            cl_command_queue queue,
            const xm::ocl::iop::ClImagePromise &in,
            const xm::ds::Color4u &hls_low,
            const xm::ds::Color4u &hls_up,
            const xm::ds::Color4u &color,
            bool linear,
            int mask_size, // 256, 512, ...
            int blur, // 3, 5, 7, 9, 11, ...
            int fine, // 3, 5, 7, 9, 11, ...
            int refine // 0, 1, 2, ...
    );

    xm::ocl::iop::ClImagePromise chroma_key(
            const xm::ocl::iop::ClImagePromise &in,
            const xm::ds::Color4u &hls_low,
            const xm::ds::Color4u &hls_up,
            const xm::ds::Color4u &color,
            bool linear,
            int mask_size, // 256, 512, ...
            int blur, // 3, 5, 7, 9, 11, ...
            int fine, // 3, 5, 7, 9, 11, ...
            int refine, // 0, 1, 2, ...
            int queue_index = -1
    );

    xm::ocl::iop::ClImagePromise chroma_key_single_pass(
            const xm::ocl::iop::ClImagePromise &in,
            const xm::ds::Color4u &hls_low,
            const xm::ds::Color4u &hls_up,
            const xm::ds::Color4u &color,
            bool linear,
            int mask_size, // 256, 512, ...
            int blur, // 3, 5, 7, 9, 11, ...
            int queue_index = -1
    );

    xm::ocl::iop::ClImagePromise chroma_key_single_pass(
            cl_command_queue queue,
            const xm::ocl::iop::ClImagePromise &in,
            const xm::ds::Color4u &hls_low,
            const xm::ds::Color4u &hls_up,
            const xm::ds::Color4u &color,
            bool linear,
            int mask_size, // 256, 512, ...
            int blur // 3, 5, 7, 9, 11, ...
    );

    xm::ocl::iop::ClImagePromise flip_rotate(
            const xm::ocl::iop::ClImagePromise &in,
            bool flip_x,
            bool flip_y,
            bool rotate,
            int queue_index = -1
    );

    xm::ocl::iop::ClImagePromise flip_rotate(
            cl_command_queue queue,
            const xm::ocl::iop::ClImagePromise &in,
            bool flip_x,
            bool flip_y,
            bool rotate
    );

}

#endif //XMOTION_OCL_FILTERS_H
