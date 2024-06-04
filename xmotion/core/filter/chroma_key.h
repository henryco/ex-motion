//
// Created by henryco on 6/2/24.
//

#ifndef XMOTION_CHROMA_KEY_H
#define XMOTION_CHROMA_KEY_H

#include "i_filter.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::chroma {

    using Conf = struct Conf {
        /**
         * Normalized [0-1] HSL range
         */
        cv::Scalar range;

        /**
         * New background color (BGR)
         */
        cv::Scalar color;

        /**
         * Chromakey color (BGR)
         */
        cv::Scalar key;

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
    };

    class ChromaKey : public xm::Filter {

        static inline const auto log =
                spdlog::stdout_color_mt("filter_chroma_key");

    private:
        cv::Scalar hls_key_lower, hls_key_upper;
        cv::Scalar bgr_bg_color;

        int mask_iterations = 0;
        int mask_size = 0;
        int blur_kernel = 0;
        int fine_kernel = 0;
        bool ready = false;

    public:
        void init(const Conf &conf);

        cv::Mat filter(const cv::Mat &in) override;

        cv::UMat filter(const cv::UMat &in) override;

    };

} // xm

#endif //XMOTION_CHROMA_KEY_H
