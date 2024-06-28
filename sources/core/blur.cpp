//
// Created by henryco on 28/06/24.
//

#include "../../xmotion/core/filter/blur.h"
#include "../../xmotion/core/ocl/ocl_filters.h"

namespace xm::filters {

    void Blur::reset() { /*void*/ }

    void Blur::init(int power) {
        log->debug("init blur: {}", power);
        kernel_size = (power * 2) + 1;
        initialized = true;
    }

    xm::ocl::iop::ClImagePromise Blur::filter(const ocl::iop::ClImagePromise &in, int q_idx) {
        if (!ready)
            return in;

        if (!initialized)
            throw std::logic_error("Filter is not initialized");

        return xm::ocl::blur(in, kernel_size, q_idx);
    }

    void Blur::start() {
        ready = true;
    }

    void Blur::stop() {
        ready = false;
    }
}