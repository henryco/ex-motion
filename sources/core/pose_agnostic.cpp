//
// Created by henryco on 14/07/24.
//

#include "../../xmotion/core/dnn/pose_agnostic.h"

namespace eox::dnn::pose {

    std::chrono::nanoseconds eox::dnn::pose::PoseAgnostic::timestamp() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch());
    }

    void eox::dnn::pose::PoseAgnostic::init(eox::dnn::pose::PoseInput _config) {
        config = _config;
        // TODO
    }

}
