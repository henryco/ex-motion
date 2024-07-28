//
// Created by henryco on 5/21/24.
//

#include "../../xmotion/core/algo/pose.h"

void xm::Pose::start() {
    stop();
    results.error = false;
    active = true;
}

void xm::Pose::stop() {
    active = false;
    release();
}

void xm::Pose::release() {
}

bool xm::Pose::is_active() const {
    return active;
}

const std::vector<xm::ocl::Image2D> &xm::Pose::frames() const {
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