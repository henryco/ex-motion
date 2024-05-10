//
// Created by henryco on 4/22/24.
//

#include "../xmotion/algo/triangulation.h"

void xm::Triangulation::init(const xm::nview::Initial &params) {
    config = params;
    // TODO?
}

xm::Triangulation& xm::Triangulation::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());
        for (auto &img: _frames)
            images.push_back(img.clone());
        return *this;
    }

    // TODO

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
