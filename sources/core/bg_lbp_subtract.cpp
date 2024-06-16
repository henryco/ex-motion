//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/filter/bg_lbp_subtract.h"

namespace xm::filters {

    void BgLbpSubtract::init(const bgs::Conf &conf) {

    }

    xm::ocl::iop::ClImagePromise BgLbpSubtract::filter(const ocl::Image2D &in, int q_idx) {
        return xm::ocl::iop::ClImagePromise();
    }

}