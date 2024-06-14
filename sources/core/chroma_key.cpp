//
// Created by henryco on 6/2/24.
//

#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm::chroma {

    void ChromaKey::init(const Conf &conf) {
        const auto key_hls = xm::ocv::bgr_to_hls(conf.key);

        const int bot_h = key_hls[0] - (int) (conf.range[0] * 255.f);
        const int bot_s = std::clamp(key_hls[2] - (int) (conf.range[1] * 255.f), 0, 255);
        const int bot_l = std::clamp(key_hls[1] - (int) (conf.range[2] * 255.f), 0, 255);

        const int top_h = key_hls[0] + (int) (conf.range[0] * 255.f);
        const int top_s = std::clamp(key_hls[2] + (int) (conf.range[1] * 255.f), 0, 255);
        const int top_l = std::clamp(key_hls[1] + (int) (conf.range[2] * 255.f), 0, 255);

        hls_key_lower = cv::Scalar(bot_h < 0 ? 255 + bot_h : bot_h, bot_l, bot_s);
        hls_key_upper = cv::Scalar(top_h > 255 ? top_h - 255 : top_h, top_l, top_s);

        linear_interpolation = conf.linear;
        mask_size = (conf.power + 1) * 256;
        blur_kernel = (conf.blur * 2) + 1;
        fine_kernel = std::max(3, (conf.fine * 2) + 1);
        mask_iterations = conf.refine;
        bgr_bg_color = conf.color;

        log->info("L: {}, {}, {}", hls_key_lower[0], hls_key_lower[1], hls_key_lower[2]);
        log->info("U: {}, {}, {}", hls_key_upper[0], hls_key_upper[1], hls_key_upper[2]);

        ready = true;
    }

    xm::ocl::iop::ClImagePromise ChromaKey::filter(const ocl::Image2D &in) {
        if (!ready)
            throw std::logic_error("Filter is not initialized");
        if (mask_iterations > 0 && fine_kernel >= 3)
            return xm::ocl::chroma_key(
                    in,
                    hls_key_lower,
                    hls_key_upper,
                    bgr_bg_color,
                    linear_interpolation,
                    mask_size,
                    blur_kernel,
                    fine_kernel,
                    mask_iterations);
        return xm::ocl::chroma_key_single_pass(
                in,
                hls_key_lower,
                hls_key_upper,
                bgr_bg_color,
                linear_interpolation,
                mask_size,
                blur_kernel);
    }

} // xm