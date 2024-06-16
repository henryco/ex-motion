//
// Created by henryco on 6/2/24.
//

#include "../../xmotion/core/filter/chroma_key.h"
#include "../../xmotion/core/ocl/ocl_filters.h"
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm::filters {

    void ChromaKey::init(const chroma::Conf &conf) {
        const auto key_hls = xm::ocv::bgr_to_hls(conf.key);

        const int bot_h = (int) key_hls.h - (int) conf.range.h;
        const int bot_s = std::clamp((int) key_hls.s - (int) conf.range.s, 0, 255);
        const int bot_l = std::clamp((int) key_hls.l - (int) conf.range.l, 0, 255);

        const int top_h = (int) key_hls.h + (int) conf.range.h;
        const int top_s = std::clamp((int) key_hls.s + (int) conf.range.s, 0, 255);
        const int top_l = std::clamp((int) key_hls.l + (int) conf.range.l, 0, 255);

        hls_key_lower = xm::ds::Color4u::hls(bot_h < 0 ? 255 + bot_h : bot_h, bot_l, bot_s);
        hls_key_upper = xm::ds::Color4u::hls(top_h > 255 ? top_h - 255 : top_h, top_l, top_s);

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

    xm::ocl::iop::ClImagePromise ChromaKey::filter(const ocl::Image2D &in, int q_idx) {
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
                    mask_iterations,
                    q_idx);
        return xm::ocl::chroma_key_single_pass(
                in,
                hls_key_lower,
                hls_key_upper,
                bgr_bg_color,
                linear_interpolation,
                mask_size,
                blur_kernel,
                q_idx);
    }

} // xm