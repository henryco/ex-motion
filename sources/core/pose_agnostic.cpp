//
// Created by henryco on 14/07/24.
//

#include <opencv2/core/ocl.hpp>
#include "../../xmotion/core/pose/pose_agnostic.h"
#include "../../xmotion/core/ocl/cl_kernel.h"

namespace eox::dnn::pose {

    void PoseAgnostic::reset() {
        if (roi_body_heuristics != nullptr)
            delete[] roi_body_heuristics;
        if (bg_filters != nullptr)
            delete[] bg_filters;
        if (velocity_filters != nullptr)
            delete[] velocity_filters;
        if (configs != nullptr)
            delete[] configs;
        if (sources != nullptr)
            delete[] sources;
        if (rois != nullptr)
            delete[] rois;

        if (_detected_bodies != nullptr)
            delete[] _detected_bodies;
        if (_detector_conf != nullptr)
            delete[] _detector_conf;
        if (_detector_queue != nullptr)
            delete[] _detector_queue;
        if (_work_frames != nullptr)
            delete[] _work_frames;
        if (_debug_infos != nullptr)
            delete[] _debug_infos;

        if (ocl_command_queue != nullptr)
            clReleaseCommandQueue(ocl_command_queue);
        if (ocl_context != nullptr)
            clReleaseContext(ocl_context);
        if (device_id != nullptr)
            clReleaseDevice(device_id);
    }

    PoseAgnostic::~PoseAgnostic() {
        reset();
    }

    std::chrono::nanoseconds eox::dnn::pose::PoseAgnostic::timestamp() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch());
    }

    void eox::dnn::pose::PoseAgnostic::init(const int n, const PoseInput *_configs) {
        reset();

        device_id         = (cl_device_id) cv::ocl::Device::getDefault().ptr();
        ocl_context       = (cl_context) cv::ocl::Context::getDefault().ptr();
        ocl_command_queue = xm::ocl::create_queue_device(
                                                         ocl_context,
                                                         device_id,
                                                         true,
                                                         false);

        rois                = new eox::dnn::RoI[n];
        configs             = new eox::dnn::pose::PoseInput[n];
        bg_filters          = new xm::filters::BgSubtract[n];
        sources             = new xm::ocl::iop::ClImagePromise[n];
        velocity_filters    = new eox::sig::VelocityFilter[n][FILTERS_DIM_SIZE];
        roi_body_heuristics = new xm::pose::roi::RoiBodyHeuristics[n];

        _work_frames        = new xm::ocl::iop::ClImagePromise[n];
        _detector_conf      = new xm::dnn::run::DetectorRoiConf[n];
        _detected_bodies    = new xm::dnn::run::DetectedBody[n];
        _debug_infos        = new PoseDebug[n];
        _detector_queue     = new int[n];

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
                velocity_filters[i][j].init(
                        config.f_win_size,
                        config.f_v_scale,
                        config.f_fps);
            }

            if (config.bgs_enable) {

                auto bgs_config           = config.bgs_config;
                bgs_config.color_channels = 3;
                bgs_config.debug_on       = false;
                bgs_config.mask_xc        = true;
                bg_filters[i].init(bgs_config);
            }
        }

        detector.init(xm::dnn::run::ModelDetector::F_32);
        marker.init(xm::dnn::run::ModelPose::HEAVY_F32);

        initialized = true;
        prediction  = false;
        n_size      = n;
    }

    void PoseAgnostic::pass(const xm::ocl::Image2D *frames, PoseResult *result, PoseDebug *debug) {
        const int &n                 = n_size;
        int        _detector_queue_n = 0;

        for (int i = 0, q = 0; i < n; i++) {
            auto &roi_body_heuristic = roi_body_heuristics[i];
            auto &frame = frames[i];
            const auto &roi = rois[i];

            if (roi_body_heuristic.get_prediction()) {
                sources[i] = xm::ocl::iop::copy_ocl(
                        frame, ocl_command_queue,
                        (int) roi.x, (int) roi.y,
                        (int) roi.w, (int) roi.h);
            } else {
                _detector_queue_n++;
                _detector_queue[q++] = i;
                _work_frames[q]      = {frame, ocl_command_queue};
                _detector_conf[q]    = {
                    .margin = configs[i].roi_margin,
                    .padding_x = configs[i].roi_padding_x,
                    .padding_y = configs[i].roi_padding_y,
                    .scale = configs[i].roi_scale
                };
            }
        }

        if (_detector_queue_n <= 0)
            goto roi_complete;

        detector.detect(_detector_queue_n, _work_frames, _detector_conf, _detected_bodies);

        for (int q = 0; q < _detector_queue_n; q++) {
            const int i = _detector_queue[q];
            const auto &detections = _detected_bodies[q];

            if (debug != nullptr) {
                _debug_infos[i].detector_score = detections.score;
            }

            if (detections.score < configs[i].threshold_detector) {
                PoseResult pose_result;
                pose_result.present = false;
                pose_result.output.score = 0;
                result[i] = pose_result;
                continue;
            }

            auto &frame = frames[i];
            auto &roi = rois[i];
            roi = detections.roi;
            sources[i] = xm::ocl::iop::copy_ocl(
                        frame, ocl_command_queue,
                        (int) roi.x, (int) roi.y,
                        (int) roi.w, (int) roi.h);
        }


        roi_complete:
        xm::ocl::iop::ClImagePromise::finalizeAll(sources, n);


        // TODO

        if (!debug)
            return;

        for (int i = 0; i < n; i++) {
            debug[i] = _debug_infos[i];
        }
    }

}
