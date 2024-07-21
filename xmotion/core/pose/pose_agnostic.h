//
// Created by henryco on 14/07/24.
//

#ifndef XMOTION_POSE_AGNOSTIC_H
#define XMOTION_POSE_AGNOSTIC_H

#include <chrono>
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "roi/roi_body_heuristics.h"

#include "../dnn/net/dnn_common.h"
#include "../filter/bg_subtract.h"
#include "../ocl/ocl_interop.h"
#include "../utils/velocity_filter.h"
#include "body/body_marker.h"
#include "detector/body_detector.h"

namespace eox::dnn::pose {

#define FILTERS_DIM_SIZE 117

    using PoseDebug = struct {
        // TODO
    };

    using PoseResult = struct {
        PoseOutput output;
        PoseDebug debug;
        long duration;
        bool present;
    };

    using PoseInput = struct PoseInput {

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

        /**
         * Whether to perform segmentation and bg subtraction
         */
        bool bgs_enable = false;

        /**
         * Background subtraction config
         */
        xm::filters::bgs::Conf bgs_config{};
    };

    class PoseAgnostic {
        static inline const auto log =
                spdlog::stdout_color_mt("pose_agnostic");

    private:
        bool prediction = false;
        bool initialized = false;

        cl_command_queue ocl_command_queue = nullptr;
        cl_device_id device_id = nullptr;
        cl_context ocl_context = nullptr;

        xm::dnn::run::BodyDetector detector;
        xm::dnn::run::BodyMarker marker;

        // ARRAYS
        xm::filters::BgSubtract *bg_filters = nullptr;
        xm::pose::roi::RoiBodyHeuristics *roi_body_heuristics = nullptr;
        eox::sig::VelocityFilter (*velocity_filters)[FILTERS_DIM_SIZE] = nullptr;
        xm::ocl::iop::ClImagePromise *sources = nullptr;
        eox::dnn::RoI *rois = nullptr;
        PoseInput *configs = nullptr;
        // ARRAYS


    public:
        void init(int n, const PoseInput *config);

        void pass(int n, const xm::ocl::Image2D *frames, PoseResult *result, PoseDebug *debug);

        ~PoseAgnostic();

    protected:
        [[nodiscard]] static std::chrono::nanoseconds timestamp();

        void reset();
    };

}

#endif //XMOTION_POSE_AGNOSTIC_H
