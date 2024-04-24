//
// Created by henryco on 4/22/24.
//

#include <opencv2/imgproc.hpp>
#include "../xmotion/algo/calibration.h"
#include "../xmotion/utils/cv_utils.h"

void xm::Calibration::init(const xm::calib::Initial &params) {
    timer.set_delay(params.delay);
    config = params;
}

bool xm::Calibration::capture_squares(const cv::Mat &frame) {
    const auto squares = xm::ocv::find_squares(
            frame,
            config.columns,
            config.rows
    );

    images.clear();
    images.push_back(squares.result);

    if (!squares.found) {
        results.remains_ms = config.delay;
        results.ready = false;
        timer.reset();
        return false;
    }

    const auto remains = timer.tick([this, &squares]() {
        image_points.push_back(squares.corners);
    });

    results.remains_ms = remains;
    results.remains_cap = config.total - (int) image_points.size();
    results.ready = results.remains_cap <= 0;

    return results.ready;
}

xm::Calibration &xm::Calibration::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());
        for (auto &img: _frames)
            images.push_back(img);
        return *this;
    }

    if (!capture_squares(_frames.front())) {
        put_debug_text();
        return *this;
    }

    // TODO CALIBRATION LOGIC

    stop();
    return *this;
}

void xm::Calibration::put_debug_text() {
    if (!DEBUG)
        return;

    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const cv::Scalar color(0, 0, 255);

    const std::string t1 = std::to_string(image_points.size()) + " / " + std::to_string(config.total);
    const std::string t2 = std::to_string(results.remains_ms) + " ms";

    cv::putText(images.front(), t1, cv::Point(20, 50), font, 1.5, color, 3);
    cv::putText(images.front(), t2, cv::Point(20, 100), font, 1.5, color, 3);
}

void xm::Calibration::start() {
    results.remains_cap = config.total;
    results.remains_ms = config.delay;
    results.ready = false;

    image_points.clear();
    active = true;
    timer.start();
}

void xm::Calibration::stop() {
    results.remains_cap = 0;
    results.remains_ms = 0;
    results.ready = true;

    image_points.clear();
    active = false;
    timer.stop();
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

void xm::Calibration::debug(bool _debug) {
    DEBUG = _debug;
}