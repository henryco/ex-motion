//
// Created by henryco on 4/22/24.
//

#include "../xmotion/algo/pose.h"
#include "../xmotion/utils/eox_globals.h"


void xm::Pose::init(const xm::nview::Initial &params) {
    results.error = false;
    config = params;
}

xm::Pose &xm::Pose::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());
        for (auto &img: _frames)
            images.push_back(img);
        return *this;
    }

    std::vector<cv::Mat> output_frames;
    std::vector<std::future<eox::dnn::PosePipelineOutput>> features;
    enqueue_inference(features, _frames, output_frames);

    std::vector<eox::dnn::PosePipelineOutput> outputs;
    if (!resolve_inference(features, outputs)) {
        stop();
        results.error = true;
        return *this;
    }

    // TODO: PROCESS RESULTS

    images.clear();
    for (const auto &frame: output_frames) {
        images.push_back(frame);
    }

    results.error = false;
    return *this;
}

void xm::Pose::enqueue_inference(std::vector<std::future<eox::dnn::PosePipelineOutput>> &io_features,
                                 const std::vector<cv::Mat> &in_frames,
                                 std::vector<cv::Mat> &out_frames) {
    io_features.reserve(config.views);
    out_frames.reserve(config.views);

    int i = 0, j = 0;
    while (i < config.views) {
        if (j >= workers.size())
            j = 0;

        const auto &frame = in_frames.at(i);
        const auto &pose = poses.at(i);

        if (DEBUG) {
            out_frames.emplace_back();
            io_features.push_back(
                    workers.at(j)->execute<eox::dnn::PosePipelineOutput>(
                            [i, frame, &pose, &out_frames]() -> eox::dnn::PosePipelineOutput {
                                cv::Mat segmented;
                                return pose->pass(frame, segmented, out_frames.at(i));
                            }));
        } else {
            out_frames.push_back(frame);
            io_features.push_back(
                    workers.at(j)->execute<eox::dnn::PosePipelineOutput>(
                            [i, frame, &pose]() -> eox::dnn::PosePipelineOutput {
                                cv::Mat segmented;
                                return pose->pass(frame, segmented);
                            }));
        }

        i++;
        j++;
    }
}

bool xm::Pose::resolve_inference(std::vector<std::future<eox::dnn::PosePipelineOutput>> &in_futures,
                                 std::vector<eox::dnn::PosePipelineOutput> &out_results) {
    for (auto &feature: in_futures) {
        if (!feature.valid())
            return false;
        if (feature.wait_for(std::chrono::milliseconds(eox::globals::TIMEOUT_MS)) == std::future_status::timeout)
            return false;
        try {
            out_results.push_back(feature.get());
        } catch (const std::exception &e) {
            return false;
        }
    }
    return true;
}

void xm::Pose::start() {
    stop();

    for (int i = 0; i < config.views; ++i) {
        auto p = std::make_unique<eox::dnn::PosePipeline>();
        p->enableSegmentation(config.segmentation);
        p->setBodyModel(config.body_model);
        p->setDetectorModel(config.detector_model);
        p->setDetectorThreshold(config.threshold_detector);
        p->setPresenceThreshold(config.threshold_presence);
        p->setPoseThreshold(config.threshold_pose);
        p->setFilterVelocityScale(config.filter_velocity_factor);
        p->setFilterWindowSize(config.filter_windows_size);
        p->setFilterTargetFps(config.filter_target_fps);
        p->setRoiRollbackWindow(config.roi_rollback_window);
        p->setRoiPredictionWindow(config.roi_center_window);
        p->setRoiClampWindow(config.roi_clamp_window);
        p->setRoiScale(config.roi_scale);
        p->setRoiMargin(config.roi_margin);
        p->setRoiPaddingX(config.roi_padding_x);
        p->setRoiPaddingY(config.roi_padding_y);
        poses.push_back(std::move(p));
    }

    for (int i = 0; i < config.threads; ++i) {
        auto p = std::make_unique<eox::util::ThreadPool>();
        p->start(1);
        workers.push_back(std::move(p));
    }

    results.error = false;
    active = true;
}

void xm::Pose::stop() {
    active = false;
    release();
}

void xm::Pose::release() {
    for (auto &worker: workers)
        worker->shutdown();
    workers.clear();
    poses.clear();
}

bool xm::Pose::is_active() const {
    return active;
}

const std::vector<cv::Mat> &xm::Pose::frames() const {
    return images;
}

const xm::nview::Result &xm::Pose::result() const {
    return results;
}

void xm::Pose::debug(bool _debug) {
    DEBUG = _debug;
}

xm::Pose::~Pose() {
    release();
}
