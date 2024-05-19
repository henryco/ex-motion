//
// Created by henryco on 4/22/24.
//

#include <opencv2/calib3d.hpp>
#include "../../xmotion/core/algo/pose.h"
#include "../../xmotion/core/utils/eox_globals.h"


void xm::Pose::init(const xm::nview::Initial &params) {
    results.error = false;
    config = params;
}

xm::Pose &xm::Pose::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());

        if (_frames.size() != config.devices.size()) {
            results.error = true;
            return *this;
        }

        for (int i = 0; i < _frames.size(); i++)
            images.push_back(undistorted(_frames.at(i), i));
        return *this;
    }

    std::vector<cv::Mat> input_frames;
    input_frames.reserve(_frames.size());
    for (int i = 0; i < _frames.size(); i++)
        input_frames.push_back(undistorted(_frames.at(i), i));

    std::vector<cv::Mat> output_frames;
    std::vector<std::future<eox::dnn::PosePipelineOutput>> features;
    enqueue_inference(features, input_frames, output_frames);

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
    io_features.reserve(config.devices.size());
    out_frames.reserve(config.devices.size());

    int i = 0, j = 0;
    while (i < config.devices.size()) {
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


cv::Mat xm::Pose::undistorted(const cv::Mat &in, int index) {
    if (!config.devices.at(index).undistort_source)
        return in;
    const auto &maps = remap_maps.at(index);
    cv::Mat undistorted;
    cv::remap(in, undistorted, maps.map1, maps.map2, cv::INTER_LINEAR);
    return std::move(undistorted);
}

void xm::Pose::start() {
    stop();

    for (const auto &device: config.devices) {
        auto p = std::make_unique<eox::dnn::PosePipeline>();
        p->enableSegmentation(config.segmentation);
        p->setBodyModel(device.body_model);
        p->setDetectorModel(device.detector_model);
        p->setDetectorThreshold(device.threshold_detector);
        p->setMarksThreshold(device.threshold_marks);
        p->setPoseThreshold(device.threshold_pose);
        p->setRoiThreshold(device.threshold_roi);
        p->setFilterVelocityScale(device.filter_velocity_factor);
        p->setFilterWindowSize(device.filter_windows_size);
        p->setFilterTargetFps(device.filter_target_fps);
        p->setRoiRollbackWindow(device.roi_rollback_window);
        p->setRoiPredictionWindow(device.roi_center_window);
        p->setRoiClampWindow(device.roi_clamp_window);
        p->setRoiScale(device.roi_scale);
        p->setRoiMargin(device.roi_margin);
        p->setRoiPaddingX(device.roi_padding_x);
        p->setRoiPaddingY(device.roi_padding_y);
        poses.push_back(std::move(p));
    }

    for (int i = 0; i < config.threads; ++i) {
        auto p = std::make_unique<eox::util::ThreadPool>();
        p->start(1);
        workers.push_back(std::move(p));
    }

    remap_maps.clear();
    remap_maps.reserve(config.devices.size());
    for (const auto &device: config.devices) {
        auto im_size = cv::Size(device.width, device.height);
        auto new_mat = cv::getOptimalNewCameraMatrix(
                device.K,
                device.D,
                im_size,
                device.undistort_alpha);
        cv::Mat map_1, map_2;
        cv::initUndistortRectifyMap(
                device.K,
                device.D,
                cv::Mat(),
                new_mat,
                im_size,
                CV_16SC2,
                map_1,
                map_2);
        remap_maps.emplace_back(
                new_mat,
                map_1,
                map_2);
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
