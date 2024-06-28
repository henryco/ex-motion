//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_BLUR_H
#define XMOTION_BLUR_H

#include "i_filter.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm::filters {

    class Blur : public xm::Filter {

        static inline const auto log =
                spdlog::stdout_color_mt("filter_blur");

    private:
        int kernel_size = 9;
        bool ready = false;
        bool initialized = false;

    public:
        void init(int power);

        xm::ocl::iop::ClImagePromise filter(const ocl::iop::ClImagePromise &in, int q_idx) override;

        void reset() override;

        void start() override;

        void stop() override;
    };
}

#endif //XMOTION_BLUR_H
