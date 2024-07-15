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

        initialized = true;
        prediction = false;
    }

    PoseResult PoseAgnostic::pass(const xm::ocl::iop::ClImagePromise &input) {
        const auto t0 = std::chrono::system_clock::now();

        const bool first_run = !initialized;

        if (!initialized) {
            init(config);
        }

        PoseResult output;


        const auto t1 = std::chrono::system_clock::now();

        output.duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        return output;
    }

}
