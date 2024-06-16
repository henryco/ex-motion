//
// Created by henryco on 16/06/24.
//

#ifndef XMOTION_BG_LBP_SUBTRACT_H
#define XMOTION_BG_LBP_SUBTRACT_H

#include "../utils/xm_data.h"
#include "i_filter.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::filters {

    namespace bgs {
        using Conf = struct Conf {
            /**
             * New background color (BGR)
             */
            xm::ds::Color4u color;

            /**
             * Local Binary Patterns similarity cutoff threshold [0...1]
             */
            float threshold = 0;

            /**
             * Delay in starting filter (ms)
             */
            long delay = 0;

            /**
             * Local Binary Patterns windows size
             *
             * \code
             * (CxC): C = clamp[ (window * 2) + 1,  3,  15]
             * \endcode
             */
            int window = 0;

            /**
             * Mask refinement iterations
             */
            int refine = 0;

            /**
             * Mask refinement kernel
             *
             * \code
             * (CxC): C = max[3, (fine * 2) + 1]
             * \endcode
             */
            int fine = 0;

            /**
             * Blur intensity,
             * used to calculate gaussian blur kernel
             *
             * \code
             * (CxC): C = (blur * 2) + 1
             * \endcode
             */
            int blur = 0;
        };
    }

    class BgLbpSubtract : public xm::Filter {
        static inline const auto log =
                spdlog::stdout_color_mt("filter_bg_lbp_subtract");

    private:
        xm::ds::Color4u bgr_bg_color;

        std::chrono::milliseconds t0;
        ocl::Image2D reference_lbp;

        int fine_iterations = 0;
        int fine_kernel = 0;
        int blur_kernel = 0;
        int lbp_kernel = 0;
        float threshold = .5f;
        long delay;

        bool initialized = false;
        bool active = false;
        bool ready = false;

    public:
        void init(const bgs::Conf &conf);

        xm::ocl::iop::ClImagePromise filter(const ocl::Image2D &in, int q_idx) override;

        void reset() override;
    };
}

#endif //XMOTION_BG_LBP_SUBTRACT_H
