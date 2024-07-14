//
// Created by henryco on 14/07/24.
//

#ifndef XMOTION_POSE_AGNOSTIC_H
#define XMOTION_POSE_AGNOSTIC_H

#include <chrono>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "net/dnn_common.h"

namespace eox::dnn::pose {

    using PoseTimePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>;

    using PoseResult = struct {
        PoseOutput output;
        bool present;
    };

    using PoseInput = struct PoseInput {
        /**
         * Region of Interest
         */
        eox::dnn::RoI roi;

        /**
         * Margins added to ROI
         */
        float roi_margin = 0.f;

        /**
         * Horizontal paddings added to ROI
         */
        float roi_padding_x = 0.f;

        /**
         * Vertical paddings added to ROI
         */
        float roi_padding_y = 0.f;

        /**
         * Scaling factor for ROI (multiplication)
         */
        float roi_scale = 1.2f;

        /**
         * Distance between detectors and actual ROI mid point
         * for which detected ROI should be rolled back to previous one
         *
         * [0.0 ... 1.0]
         */
        float roi_rollback_window = 0.f;

        /**
         * Distance between actual and predicted ROI mid point
         * for which should stay unchanged (helps reducing jittering)
         *
         * [0.0 ... 1.0]
         */
        float roi_center_window = 0.f;

        /**
         * Acceptable ratio of clamped to original ROI size. \n
         * Zero (0) means every size is acceptable \n
         * One (1) means only original (non-clamped) ROI is acceptable \n
         *
         * [0.0 ... 1.0]
         */
        float roi_clamp_window = 0.f;

        /**
         * Threshold score for landmarks presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_marks = 0.5;

        /**
         * Threshold score for detector ROI presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_detector = 0.5;

        /**
         * Threshold score for body presence
         *
         * [0.0 ... 1.0]
         */
        float threshold_pose = 0.5;

        /**
         * Threshold score for detector ROI distance to body marks. \n
         * In other works: How far marks should be from detectors ROI borders. \n
         * Currently implemented only for horizontal axis.
         *
         * [0.0 ... 1.0]
         */
        float threshold_roi = 0.f;

        /**
         * Low-pass filter velocity scale: lower -> smoother, but adds lag.
         */
        float f_v_scale = 0.5;

        /**
         * Low-pass filter window size: higher -> smoother, but adds lag.
         */
        int f_win_size = 30;

        /**
         * Low-pass filter target fps.
         * Important to properly calculate points movement speed.
         */
        int f_fps = 30;
    };

    class PoseAgnostic {
        static inline const auto log =
                spdlog::stdout_color_mt("pose_agnostic");

    private:
        bool preserved_roi = false;
        bool discarded_roi = false;
        bool rollback_roi = false;
        bool prediction = false;
        bool initialized = false;

        PoseInput config;

    public:
        void init(PoseInput config);
        // TODO

    protected:
        [[nodiscard]] std::chrono::nanoseconds timestamp() const;
    };

}

#endif //XMOTION_POSE_AGNOSTIC_H
