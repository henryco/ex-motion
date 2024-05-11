//
// Created by henryco on 4/22/24.
//

#include "../xmotion/algo/triangulation.h"

void xm::Triangulation::init(const xm::nview::Initial &params) {
    config = params;

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

    // TODO
}

xm::Triangulation &xm::Triangulation::proceed(float delta, const std::vector<cv::Mat> &_frames) {
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

        const auto& frame = _frames.at(i);
        frames.emplace_back();
        features.push_back(
                workers.at(j)->execute<eox::dnn::PosePipelineOutput>([this, i, frame, &frames]() -> eox::dnn::PosePipelineOutput {
                    cv::Mat segmented;
                    return poses.at(i)->pass(frame, segmented, frames.at(i));
                }));

        i++;
        j++;
    }

    for (auto &feature: features) {
        const auto result = feature.get();
        // TODO: PROCESS RESULTS
    }

    images.clear();
    for (const auto &frame: frames) {
        images.push_back(frame);
    }

    return *this;
}

void xm::Triangulation::start() {
    active = true;
}

void xm::Triangulation::stop() {
    active = false;
}

bool xm::Triangulation::is_active() const {
    return active;
}

const std::vector<cv::Mat> &xm::Triangulation::frames() const {
    return images;
}

const xm::nview::Result &xm::Triangulation::result() const {
    return results;
}

void xm::Triangulation::debug(bool _debug) {
    DEBUG = _debug;
}
