//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/filter/bg_lbp_subtract.h"
#include "../../xmotion/core/ocl/ocl_filters.h"

namespace xm::filters {

    void BgLbpSubtract::reset() {
        t0 = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch());
        ready = false;
    }

    void BgLbpSubtract::init(const bgs::Conf &conf) {
        threshold = std::clamp(conf.threshold, 0.f, 1.f);
        lbp_kernel = std::clamp(conf.window * 2 + 1, 3, 15);
        blur_kernel = std::max(0, conf.blur * 2 + 1);
        fine_kernel = std::max(3, conf.fine * 2 + 1);
        fine_iterations = std::max(0, conf.refine);
        delay = std::max(0L, conf.delay);
        bgr_bg_color = conf.color;

        initialized = true;
        reset();
    }

    xm::ocl::iop::ClImagePromise BgLbpSubtract::filter(const ocl::Image2D &frame_in, int q_idx) {
        if (!initialized)
            throw std::logic_error("Filter is not initialized");

        if (!ready) {
            const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch());
            if (now - t0 >= std::chrono::milliseconds(delay)) {
//                reference_buffer = frame_in;
                reference_buffer = xm::ocl::blur(
                        xm::ocl::blur(
                                xm::ocl::blur(
                                        frame_in,
                                        21,
                                        q_idx),
                                21,
                                q_idx),
                        21,
                        q_idx)
                                .waitFor()
                                .getImage2D();
                ready = true;
            }
            return frame_in;
        }

        // TODO mask refinement
        return xm::ocl::subtract_bg_color_diff(
                reference_buffer,
                xm::ocl::blur(
                        xm::ocl::blur(
                                xm::ocl::blur(
                                        frame_in,
                                        21,
                                        q_idx),
                                21,
                                q_idx),
                        21,
                        q_idx),
                bgr_bg_color,
                threshold,
                q_idx);
    }

}