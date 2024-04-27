//
// Created by henryco on 4/27/24.
//

#include "../xmotion/algo/cross.h"

void xm::CrossCalibration::init(const xm::cross::Initial &params) {
    timer.set_delay(params.delay);
    config = params;
}

bool xm::CrossCalibration::capture_squares(const std::vector<cv::Mat> &_frames) {
    return false;
}

void xm::CrossCalibration::calibrate() {

    // Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    std::vector<cv::Point3f> obj_p;
    for (int i = 0; i < config.rows - 1; ++i) {
        for (int j = 0; j < config.columns - 1; ++j) {
            obj_p.emplace_back((float) j, (float) i, 0.0f);
        }
    }

    // TODO
}

xm::CrossCalibration &xm::CrossCalibration::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());
        for (auto &img: _frames)
            images.push_back(img);
        return *this;
    }

    if (!capture_squares(_frames)) {
        put_debug_text();
        return *this;
    }

    calibrate();

    return *this;
}

void xm::CrossCalibration::put_debug_text() {
    if (!DEBUG)
        return;
    // TODO
}

void xm::CrossCalibration::start() {
    results.remains_cap = config.total;
    results.remains_ms = config.delay;
    results.ready = false;

//    image_points.clear();
    active = true;
    timer.start();
}

void xm::CrossCalibration::stop() {
    results.remains_cap = 0;
    results.remains_ms = 0;
    results.ready = false;

//    image_points.clear();
    active = false;
    timer.stop();
}

bool xm::CrossCalibration::is_active() const {
    return active;
}

const std::vector<cv::Mat> &xm::CrossCalibration::frames() const {
    return images;
}

const xm::cross::Result &xm::CrossCalibration::result() const {
    return results;
}

void xm::CrossCalibration::debug(bool _debug) {
    DEBUG = _debug;
}
