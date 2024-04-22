//
// Created by henryco on 4/22/24.
//

#include "../xmotion/algo/calibration.h"

xm::Calibration& xm::Calibration::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    images.clear();
    images.reserve(_frames.size());
    for (auto &img: _frames)
        images.push_back(img);

    if (!is_active())
        return *this;

    // TODO LOGIC HERE

    return *this;
}

void xm::Calibration::start() {
    active = true;
}

void xm::Calibration::stop() {
    active = false;
}

bool xm::Calibration::is_active() const {
    return active;
}

const std::vector<cv::Mat> &xm::Calibration::frames() const {
    return images;
}

const xm::calib::Result &xm::Calibration::result() const {
    return results;
}
