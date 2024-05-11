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

    std::vector<std::future<eox::dnn::PosePipelineOutput>> features;
    std::vector<cv::Mat> frames;
    features.reserve(config.views);
    frames.reserve(config.views);

    int i = 0, j = 0;
    while (i < config.views) {
        if (j >= workers.size())
            j = 0;

        const auto &frame = _frames.at(i);
        const auto &pose = poses.at(i);
        frames.emplace_back();
        features.push_back(
                workers.at(j)->execute<eox::dnn::PosePipelineOutput>(
                        [i, frame, &pose, &frames]() -> eox::dnn::PosePipelineOutput {
                            cv::Mat segmented;
                            return pose->pass(frame, segmented, frames.at(i));
                        }));

        i++;
        j++;
    }

    for (auto &feature: features) {
        eox::dnn::PosePipelineOutput result;

        if (!feature.valid()) {
            results.error = true;
            stop();
            return *this;
        }

        if (feature.wait_for(std::chrono::milliseconds(eox::globals::TIMEOUT_MS)) == std::future_status::timeout) {
            results.error = true;
            stop();
            return *this;
        }

        try {
            result = feature.get();
        } catch (const std::exception &e) {
            results.error = true;
            stop();
            return *this;
        }

        // TODO: PROCESS RESULTS
    }

    images.clear();
    for (const auto &frame: frames) {
        images.push_back(frame);
    }

    results.error = false;
    return *this;
}

void xm::Pose::start() {
    stop();

    for (int i = 0; i < config.views; ++i) {
        auto p = std::make_unique<eox::dnn::PosePipeline>();
        p->enableSegmentation(config.segmentation);
        p->setBodyModel(eox::dnn::pose::FULL_F32);
        p->setDetectorModel(eox::dnn::box::F_16);
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
