//
// Created by henryco on 6/2/24.
//

#ifndef XMOTION_CHROMA_KEY_H
#define XMOTION_CHROMA_KEY_H

#include "i_filter.h"
#include "../utils/xm_data.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::filters {

    namespace chroma {
        using Conf = struct Conf {
            /**
             * Normalized [0-1] HSL range
             */
            xm::ds::Color4u range;

            /**
             * New background color (BGR)
             */
            xm::ds::Color4u color;

            /**
             * Chromakey color (BGR)
             */
            xm::ds::Color4u key;

            /**
             * Mask refinement iterations
             */
            int refine = 0;

            /**
             * Mask refinement kernel
             *
             * \code
             * (CxC): C = max(3, (fine * 2) + 1)
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

            /**
             * Mask size, multiple of 256
             *
             * \code
             * (TxT): T = (1 + power) * 256
             * \endcode
             */
            int power = 0;

            /**
             * Should use linear interpolation
             * (mask is slower but smoother)
             */
            bool linear = false;
        };
    }

    class ChromaKey : public xm::Filter {

        static inline const auto log =
                spdlog::stdout_color_mt("filter_chroma_key");

    private:
        xm::ds::Color4u hls_key_lower;
        xm::ds::Color4u hls_key_upper;
        xm::ds::Color4u bgr_bg_color;

        bool linear_interpolation = false;
        int mask_iterations = 0;
        int mask_size = 0;
        int blur_kernel = 0;
        int fine_kernel = 0;
        bool ready = false;

    public:
        void init(const chroma::Conf &conf);

        xm::ocl::iop::ClImagePromise filter(const ocl::iop::ClImagePromise &in, int q_idx) override;

        void reset() override;
    };

} // xm

#endif //XMOTION_CHROMA_KEY_H
