//
// Created by henryco on 14/07/24.
//

#include "../../xmotion/core/pose/pose_agnostic.h"

namespace eox::dnn::pose {

    void PoseAgnostic::reset() {
        if (roi_body_heuristics != nullptr)
            delete[] roi_body_heuristics;
        if (velocity_filters != nullptr)
            delete[] velocity_filters;
        if (configs != nullptr)
            delete[] configs;
    }

    PoseAgnostic::~PoseAgnostic() {
        reset();
    }

    std::chrono::nanoseconds eox::dnn::pose::PoseAgnostic::timestamp() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch());
    }

    void eox::dnn::pose::PoseAgnostic::init(int n, const PoseInput *_configs) {
        reset();

        configs = new PoseInput[n];
        velocity_filters = new eox::sig::VelocityFilter[n][FILTERS_DIM_SIZE];
        roi_body_heuristics = new xm::pose::roi::RoiBodyHeuristics[n];

        for (int i = 0; i < n; i++) {
            const auto &config = _configs[i];
            configs[i] = config;

            roi_body_heuristics[i].margin = config.roi_margin;
            roi_body_heuristics[i].padding_x = config.roi_padding_x;
            roi_body_heuristics[i].padding_y = config.roi_padding_y;
            roi_body_heuristics[i].scale = config.roi_scale;
            roi_body_heuristics[i].rollback_window = config.roi_rollback_window;
            roi_body_heuristics[i].center_window = config.roi_center_window;
            roi_body_heuristics[i].clamp_window = config.roi_clamp_window;
            roi_body_heuristics[i].threshold = config.threshold_roi;

            for (int j = 0; j < FILTERS_DIM_SIZE; j++) {
                log->info("init filter[{}][{}]", i, j);
                velocity_filters[i][j].with(
                        config.f_win_size,
                        config.f_v_scale,
                        config.f_fps);
            }

        }

        // TODO

        initialized = true;
        prediction = false;
    }

    void PoseAgnostic::pass(int n, xm::ocl::Image2D *frames, PoseResult *result, PoseDebug *debug) {

    }

}
