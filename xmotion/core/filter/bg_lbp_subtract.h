//
// Created by henryco on 16/06/24.
//

#ifndef XMOTION_BG_LBP_SUBTRACT_H
#define XMOTION_BG_LBP_SUBTRACT_H

#include "../utils/xm_data.h"
#include "i_filter.h"

#include <map>
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
        // ===== OCL PART =====
        std::map<int, cl_command_queue> ocl_queue_map;
        cl_command_queue ocl_command_queue;
        cl_device_id device_id;
        cl_context ocl_context;
        // ===== OCL PART =====



        xm::ds::Color4u bgr_bg_color;
//        ocl::Image2D reference_buffer;

        int fine_iterations = 0;
        int fine_kernel = 0;
        int blur_kernel = 0;
        int lbp_kernel = 0;
        float threshold = .5f;
        long delay = 0;

        bool initialized = false;
        bool ready = false;


    public:
        BgLbpSubtract();

        ~BgLbpSubtract() override;

        void init(const bgs::Conf &conf);

        xm::ocl::iop::ClImagePromise filter(const ocl::Image2D &in, int q_idx) override;

        void reset() override;

    protected:
        cl_command_queue  retrieve_queue(int index);
    };
}

#endif //XMOTION_BG_LBP_SUBTRACT_H
