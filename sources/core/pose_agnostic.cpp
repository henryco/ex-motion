//
// Created by henryco on 14/07/24.
//

#include <opencv2/core/ocl.hpp>
#include "../../xmotion/core/pose/pose_agnostic.h"
#include "../../xmotion/core/ocl/cl_kernel.h"

namespace eox::dnn::pose {

    void PoseAgnostic::reset() {
        if (roi_body_heuristics)
            delete[] roi_body_heuristics;
        if (bg_filters)
            delete[] bg_filters;
        if (velocity_filters)
            delete[] velocity_filters;
        if (configs)
            delete[] configs;
        if (sources)
            delete[] sources;
        if (rois)
            delete[] rois;

        if (_detected_bodies)
            delete[] _detected_bodies;
        if (_detector_conf)
            delete[] _detector_conf;
        if (_detector_queue)
            delete[] _detector_queue;
        if (_work_frames)
            delete[] _work_frames;
        if (_debug_infos)
            delete[] _debug_infos;
        if (_work_metadata)
            delete[] _work_metadata;
        if (_pose_outputs)
            delete[] _pose_outputs;
        if (_pose_results)
            delete[] _pose_results;
        if (_pose_queue)
            delete[] _pose_queue;

        if (ocl_command_queues != nullptr) {
            for (int i = 0; i < n_size; i++)
                clReleaseCommandQueue(ocl_command_queues[i]);
            delete[] ocl_command_queues;
        }
        if (ocl_context)
            clReleaseContext(ocl_context);
        if (device_id)
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

        ocl_command_queues  = new cl_command_queue[n];
        rois                = new eox::dnn::RoI[n];
        configs             = new eox::dnn::pose::PoseInput[n];
        bg_filters          = new xm::filters::BgSubtract[n];
        sources             = new xm::ocl::iop::ClImagePromise[n];
        velocity_filters    = new eox::sig::VelocityFilter[n][FILTERS_DIM_SIZE];
        roi_body_heuristics = new xm::pose::roi::RoiBodyHeuristics[n];
        _work_frames        = new xm::ocl::iop::ClImagePromise[n];
        _detector_conf      = new xm::dnn::run::DetectorRoiConf[n];
        _detected_bodies    = new xm::dnn::run::DetectedBody[n];
        _pose_outputs       = new eox::dnn::PoseOutput[n];
        _pose_results       = new PoseResult[n];
        _work_metadata      = new PoseWorking[n];
        _debug_infos        = new PoseDebug[n];
        _detector_queue     = new int[n];
        _pose_queue         = new int[n];

        for (int i = 0; i < n; i++) {
            ocl_command_queues[i] = xm::ocl::create_queue_device(
                                                                 ocl_context,
                                                                 device_id,
                                                                 true,
                                                                 false);

            const auto config = _configs[i];
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

    void PoseAgnostic::pass(const xm::ocl::Image2D *frames, PoseResult *results, PoseDebug *debug, long &duration) {
        int        _detector_queue_n = 0;
        int        _pose_queue_n     = 0;

        prepare_input(frames, _detector_queue_n, _pose_queue_n, debug);

        for (int i = 0; i < _pose_queue_n; i++) {
            _work_frames[i] = sources[_pose_queue[i]];
        }

        // TODO: bg removal and segmentation

        const auto t0 = std::chrono::system_clock::now();
        marker.inference(_pose_queue_n, _work_frames, _pose_outputs, false);
        const auto t1 = std::chrono::system_clock::now();

        for (int y = 0; y < _pose_queue_n; y++) {
            const auto i = _pose_queue[y];
            pose_process(i, (int) frames[i].cols, (int) frames[i].rows, debug);
        }

        duration = duration_cast<std::chrono::nanoseconds>((t1 - t0)).count();
        prepare_results(results, debug);
    }

    void PoseAgnostic::prepare_input(
        const xm::ocl::Image2D *frames,
        int &                   detector_queue_n,
        int &                   pose_queue_n,
        const bool              debug
    ) {
        const int &n = n_size;

        for (int i = 0; i < n; i++) {
            auto &      frame = frames[i];
            const auto &roi   = rois[i];

            if (_work_metadata[i].prediction) {
                sources[i] = xm::ocl::iop::copy_ocl(
                                                    frame, ocl_command_queues[i],
                                                    (int) roi.x, (int) roi.y,
                                                    (int) roi.w, (int) roi.h);

                _pose_queue[pose_queue_n++] = i;
            }
            else {
                int &q = detector_queue_n;
                _detector_queue[q] = i;
                _work_frames[q]      = { frame, ocl_command_queues[i], true };
                _detector_conf[q]    = {
                    .margin = configs[i].roi_margin,
                    .padding_x = configs[i].roi_padding_x,
                    .padding_y = configs[i].roi_padding_y,
                    .scale = configs[i].roi_scale
                };
                detector_queue_n++;
            }
        }

        if (detector_queue_n <= 0)
            return;

        detector.detect(detector_queue_n, _work_frames, _detector_conf, _detected_bodies);

        for (int q = 0; q < detector_queue_n; q++) {
            const int   i          = _detector_queue[q];
            const auto &detections = _detected_bodies[q];

            if (debug) {
                _debug_infos[i].detector_score = detections.score;
            }

            if (detections.score < configs[i].threshold_detector) {
                // nothing detected or results is just not satisfying

                _pose_results[i].present = false;
                _pose_results[i].score   = 0;
                continue;
            }

            const auto &frame = frames[i];
            auto &      roi   = rois[i];

            roi        = detections.roi;
            sources[i] = xm::ocl::iop::copy_ocl(
                                                frame, ocl_command_queues[i],
                                                (int) roi.x, (int) roi.y,
                                                (int) roi.w, (int) roi.h);

            _pose_queue[pose_queue_n++] = i;

            if (!_work_metadata[i].discarded_roi) {
                // Reset filters ONLY IF this is clear detector run (no points found previously)
                for (int h = 0; h < FILTERS_DIM_SIZE; h++)
                    velocity_filters[i][h].reset();
            }
        }
    }

    void PoseAgnostic::pose_process(
        const int  i,
        const int  width,
        const int  height,
        const bool debug
    ) {
        PoseResult &output = _pose_results[i];

        const auto  now        = timestamp();
        const auto &config     = configs[i];

        auto &      roi        = rois[i];
        auto &      work_meta  = _work_metadata[i];
        auto &      result     = _pose_outputs[i];
        auto &      heuristics = roi_body_heuristics[i];

        if (debug) {
            _debug_infos[i].pose_score = result.score;
        }

        if (result.score <= configs[i].threshold_pose && work_meta.prediction) {
            // Nothing found, retry but without prediction (clear detector run)
            work_meta.prediction    = false;
            work_meta.rollback_roi  = false;
            work_meta.discarded_roi = false;
            work_meta.preserved_roi = false;
            return; // TODO restart
        }

        work_meta.discarded_roi = false;
        work_meta.preserved_roi = false;
        work_meta.rollback_roi  = false;

        eox::dnn::Landmark landmarks[39];

        // decode landmarks and turn it back into image coordinate space (denormalize)
        for (int t = 0; t < 39; t++) {
            landmarks[t] = {
                // turning x,y into common (global) coordinates
                .x = (result.landmarks_norm[t].x * roi.w) + roi.x,
                .y = (result.landmarks_norm[t].y * roi.h) + roi.y,

                .v = result.landmarks_norm[t].v,
                .p = result.landmarks_norm[t].p,
            };
        }

        // temporal filtering (low pass based on velocity (literally))
        for (int t = 0; t < 39; t++) {
            const auto idx = t * 2;

            auto fx = velocity_filters[i][idx + 0].filter(now, landmarks[t].x);
            auto fy = velocity_filters[i][idx + 1].filter(now, landmarks[t].y);

            landmarks[t].x = fx;
            landmarks[t].y = fy;
        }

        if (config.threshold_roi > 0 && config.threshold_roi < 1) {
            const auto roi_presence_score = heuristics.presence(roi, landmarks, true);
            _debug_infos[i].roi_score     = roi_presence_score;

            if (roi_presence_score <= 0) {

                output.present = false;
                output.score   = 0;

                work_meta.prediction = false;
                // There is simply nothig, gg go next
                return;;
            }
        }

        const auto prediction = heuristics.predict(
                                                   roi,
                                                   landmarks,
                                                   width,
                                                   height);

        work_meta.preserved_roi = prediction.preserved;
        work_meta.discarded_roi = prediction.discarded;
        work_meta.rollback_roi  = prediction.rollback;
        work_meta.prediction    = prediction.prediction;
        roi                     = prediction.roi;
        output.score            = result.score;
        output.present          = true;

        memcpy(output.segmentation, result.segmentation, (size_t) 256 * 256 * sizeof(float));
        memcpy(output.landmarks, landmarks, (size_t) 39 * sizeof(eox::dnn::Landmark));
    }

    void PoseAgnostic::prepare_results(PoseResult *results, PoseDebug *debug) {
        const int &n = n_size;

        for (int i = 0; i < n; i++) {
            results[i] = _pose_results[i];
        }

        if (!debug)
            return;

        for (int i = 0; i < n; i++) {
            _debug_infos[i].preserved_roi = _work_metadata[i].preserved_roi;
            _debug_infos[i].discarded_roi = _work_metadata[i].discarded_roi;
            _debug_infos[i].rollback_roi  = _work_metadata[i].rollback_roi;
            _debug_infos[i].prediction    = _work_metadata[i].prediction;

            debug[i] = _debug_infos[i];
        }
    }

}
