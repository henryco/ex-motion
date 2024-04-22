//
// Created by henryco on 4/22/24.
//

#include "../xmotion/algo/triangulation.h"

xm::Triangulation& xm::Triangulation::proceed(float delta, const std::vector<cv::Mat> &frames) {
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
