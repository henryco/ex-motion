//
// Created by henryco on 4/27/24.
//

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "../xmotion/algo/cross.h"
#include "../xmotion/utils/cv_utils.h"

void xm::CrossCalibration::init(const xm::cross::Initial &params) {
    timer.set_delay(params.delay);
    config = params;
}

bool xm::CrossCalibration::capture_squares(const std::vector<cv::Mat> &_frames) {
    images.clear();
    std::vector<xm::ocv::Squares> squares;
    squares.reserve(_frames.size());
    for (const auto &frame: _frames) {
        const auto square = xm::ocv::find_squares(frame, config.columns, config.rows);
        images.push_back(square.result);
        squares.push_back(square);
    }

    for (const auto &square: squares) {
        if (!square.found) {
            results.remains_ms = config.delay;
            results.ready = false;
            timer.reset();
            return false;
        }
    }

    const auto remains = timer.tick([this, &squares]() {
        std::vector<std::vector<cv::Point2f>> points;
        points.reserve(squares.size());
        for (const auto &square: squares)
            points.push_back(square.corners);

        image_points.push_back(points);
    });

    results.remains_ms = remains;
    results.remains_cap = config.total - (int) image_points.size();
    results.ready = results.remains_cap <= 0;

    return results.ready;
}

void xm::CrossCalibration::calibrate() {

    // Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    std::vector<cv::Point3f> obj_p;
    for (int i = 0; i < config.rows - 1; ++i) {
        for (int j = 0; j < config.columns - 1; ++j) {
            obj_p.emplace_back((float) j, (float) i, 0.0f);
        }
    }

    // Replicate obj_p for each image
    std::vector<std::vector<cv::Point3f>> object_points;
    object_points.reserve(image_points.size());
    for (int i = 0; i < image_points.size(); ++i) {
        object_points.push_back(obj_p);
    }

    std::vector<cv::Mat> R, T, E, F;
    std::vector<double> errors;

    // cross calibration, for each pair
    for (int i = 0; i < config.views; i++) {

        // prevent cross loop calibration
        if (i == config.views - 1 && i == 1)
            break;

        const int j = (i == config.views - 1) ? 0 : (i + 1);

        const auto K1 = config.K[i];
        const auto K2 = config.K[j];
        const auto D1 = config.D[i];
        const auto D2 = config.D[j];

        cv::Mat Ri, Ti, Ei, Fi, RMSi;

        const auto corners1 = image_points[i];
        const auto corners2 = image_points[j];
        const auto rms = cv::stereoCalibrate(
                object_points,
                corners1,
                corners2,
                K1,
                D1,
                K2,
                D2,
                cv::Size(0, 0),
                Ri, Ti, Ei, Fi, RMSi,
                cv::CALIB_FIX_INTRINSIC);

        R.push_back(Ri);
        T.push_back(Ti);
        E.push_back(Ei);
        F.push_back(Fi);
        errors.push_back(rms);
    }

    active = false;
    results.R = R;
    results.T = T;
    results.E = E;
    results.F = F;
    results.mre = errors;
    results.total = (int) errors.size();

    results.ready = true;
    results.remains_ms = 0;
    results.remains_cap = 0;
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
    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    const cv::Scalar color(0, 0, 255);

    const std::string t1 = std::to_string(image_points.size()) + " / " + std::to_string(config.total);
    const std::string t2 = std::to_string(results.remains_ms) + " ms";

    cv::putText(images.front(), t1, cv::Point(20, 50), font, 1.5, color, 3);
    cv::putText(images.front(), t2, cv::Point(20, 100), font, 1.5, color, 3);
}

void xm::CrossCalibration::start() {
    results.remains_cap = config.total;
    results.remains_ms = config.delay;
    results.ready = false;

    image_points.clear();
    active = true;
    timer.start();
}

void xm::CrossCalibration::stop() {
    results.remains_cap = 0;
    results.remains_ms = 0;
    results.ready = false;

    image_points.clear();
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
