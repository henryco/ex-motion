//
// Created by henryco on 6/2/24.
//

#ifndef XMOTION_CHROMA_KEY_H
#define XMOTION_CHROMA_KEY_H

#include "a_filter.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::chroma {

    using Conf = struct {
        /**
         * Normalized [0-1] HSL range
         */
        cv::Scalar range; // h, l, s

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
        int refine;
    };

    class ChromaKey : public xm::Filter {

        static inline const auto log =
                spdlog::stdout_color_mt("filter_chroma_key");

    private:
        cv::_InputOutputArray background;

        cv::Scalar hls_key_lower, hls_key_upper;
        cv::Scalar bgr_bg_color;

        int mask_iterations = 0;
        bool up_to_date = false;
        bool ready = false;

    public:
        void init(const Conf &conf);

        void filter(cv::InputArray in, cv::OutputArray out) override;

    };

} // xm

#endif //XMOTION_CHROMA_KEY_H
