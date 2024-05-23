//
// Created by henryco on 4/27/24.
//

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "../../xmotion/core/algo/chain.h"
#include "../../xmotion/core/utils/cv_utils.h"

void xm::ChainCalibration::init(const xm::chain::Initial &params) {
    timer.set_delay(params.delay);
    config = params;

    total_pairs = config.closed ? config.views : (config.views - 1);

    image_points.clear();
    image_points.reserve(total_pairs);
}

xm::ChainCalibration &xm::ChainCalibration::proceed(float delta, const std::vector<cv::Mat> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());
        for (auto &img: _frames)
            images.push_back(img);
        put_debug_text();
        return *this;
    }

    if (!capture_squares(_frames)) {
        put_debug_text();
        return *this;
    }

    calibrate();

    return *this;
}

bool xm::ChainCalibration::capture_squares(const std::vector<cv::Mat> &_frames) {

    if (current_pair >= total_pairs) {
        results.current = 0;
        results.ready = true;
        return true;
    }

    const int left = current_pair;
    const int right = ((left + 1) >= config.views) ? 0 : (left + 1);

    images.clear();
    images.reserve(_frames.size());
    for (int i = 0; i < _frames.size(); i++) {
        if (i == left || i == right) {
            images.push_back(_frames[i]);
            continue;
        }

        {
            // graying-out inactive frames
            cv::Mat gray;
            cv::cvtColor(_frames[i], gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(gray, gray, cv::COLOR_GRAY2BGR);
            images.push_back(gray);
        }
    }

    const auto squares_l = xm::ocv::find_squares(
            _frames[left],
            config.columns,
            config.rows,
            config.sb
    );

    images[left] = squares_l.result;
    if (!squares_l.found) {
        results.current = current_pair;
        results.remains_ms = config.delay;
        results.ready = false;
        timer.reset();
        return false;
    }

    const auto squares_r = xm::ocv::find_squares(
            _frames[right],
            config.columns,
            config.rows,
            config.sb
    );

    images[right] = squares_r.result;
    if (!squares_r.found) {
        results.current = current_pair;
        results.remains_ms = config.delay;
        results.ready = false;
        timer.reset();
        return false;
    }

    const auto callback = [this, &squares_l, &squares_r]() {
        if (image_points.empty()) {
            image_points.emplace_back();
        }

        // [2: (l,r)][rows x cols][2: (x,y)]
        std::vector<std::vector<cv::Point2f>> l_r_points;
        l_r_points.push_back(squares_l.corners);
        l_r_points.push_back(squares_r.corners);

        image_points.back().push_back(l_r_points);
        counter += 1;
    };

    if (config.delay <= 0) {
        callback();
    } else {
        results.remains_ms = timer.tick(callback);
    }

    results.remains_cap = config.total - counter;
    results.current = current_pair;
    results.ready = false;

    if (results.remains_cap <= 0) {
        // All points for current pair grabbed

        current_pair += 1;
        if (current_pair >= total_pairs) {
            // Points for ALL pairs grabbed

            results.current = 0;
            results.remains_cap = 0;
            results.remains_ms = 0;
            results.ready = true;
            return true;
        }

        // Not all pairs are ready, create new pair
        image_points.emplace_back();

        // And repeat process
        counter = 0;
        results.current = current_pair;
        results.remains_cap = config.total;
        results.remains_ms = 0;
        results.ready = false;
        return false;
    }

    return false;
}

void xm::ChainCalibration::calibrate() {

    // Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    std::vector<cv::Point3f> obj_p;
    for (int i = 0; i < config.rows - 1; ++i) {
        for (int j = 0; j < config.columns - 1; ++j) {
            obj_p.emplace_back((float) j * config.size, (float) i * config.size, 0.0f);
        }
    }

    // Replicate obj_p for each image
    std::vector<std::vector<cv::Point3f>> object_points;
    object_points.reserve(image_points.back().size());
    for (int i = 0; i < image_points.back().size(); ++i) {
        object_points.push_back(obj_p);
    }

    std::vector<xm::chain::Pair> pairs;
    pairs.reserve(total_pairs);

    // cross calibration, for each pair
    for (int i = 0; i < total_pairs; i++) {
        const int j = ((i + 1) >= config.views) ? 0 : (i + 1);

        const auto K1 = config.K[i];
        const auto K2 = config.K[j];
        const auto D1 = config.D[i];
        const auto D2 = config.D[j];

        cv::Mat Ri, Ti, Ei, Fi, RMSi;

        std::vector<std::vector<cv::Point2f>> corners_l;
        std::vector<std::vector<cv::Point2f>> corners_r;

        for (const auto &item: image_points[i]) {
            corners_l.push_back(item[0]);
            corners_r.push_back(item[1]);
        }

        log->info("stereo calibrate pair: [{},{}]", i, j);
        const auto rms = cv::stereoCalibrate(
                object_points,
                corners_l,
                corners_r,
                K1,
                D1,
                K2,
                D2,
                cv::Size(0, 0),
                Ri, Ti, Ei, Fi, RMSi,
                cv::CALIB_FIX_INTRINSIC);

        cv::Mat RTi = cv::Mat::eye(4, 4, Ri.type());
        Ri.copyTo(RTi(cv::Rect(0, 0, 3, 3)));
        Ti.copyTo(RTi(cv::Rect(3, 0, 1, 3)));

        pairs.push_back({
            .R = Ri,
            .T = Ti,
            .E = Ei,
            .F = Fi,
            .RT = RTi,
            .mre = rms
        });
    }

    active = false;
    results.total = total_pairs;
    results.ready = true;
    results.current = 0;
    results.remains_ms = 0;
    results.remains_cap = 0;
    results.calibrated = pairs;
}

void xm::ChainCalibration::put_debug_text() {
    if (!DEBUG)
        return;

    if (!results.ready && !active)
        return;

    const auto font = cv::FONT_HERSHEY_SIMPLEX;

    if (results.ready && !active) {
        const cv::Scalar color(0, 255, 0);
        cv::putText(images.front(), "Calibrated", cv::Point(20, 50), font, 1.5, color, 3);
        return;
    }

    if (results.remains_ms <= 10) {
        for (auto &image: images)
            image.setTo(cv::Scalar(255, 255, 255));
    }

    const auto size = image_points.empty() ? 0 : image_points.back().size();
    const cv::Scalar color(0, 0, 255);
    const std::string t1 = std::to_string(size) + " / " + std::to_string(config.total);
    const std::string t2 = std::to_string(results.remains_ms) + " ms";

    cv::putText(images.front(), t1, cv::Point(20, 50), font, 1.5, color, 3);
    cv::putText(images.front(), t2, cv::Point(20, 100), font, 1.5, color, 3);
}

void xm::ChainCalibration::start() {
    results.current = 0;
    results.remains_cap = config.total;
    results.remains_ms = config.delay;
    results.ready = false;

    counter = 0;
    current_pair = 0;
    image_points.clear();
    active = true;
    timer.start();
}

void xm::ChainCalibration::stop() {
    results.current = 0;
    results.remains_cap = 0;
    results.remains_ms = 0;
    results.ready = false;

    counter = 0;
    current_pair = 0;
    image_points.clear();
    active = false;
    timer.stop();
}

bool xm::ChainCalibration::is_active() const {
    return active;
}

const std::vector<cv::Mat> &xm::ChainCalibration::frames() const {
    return images;
}

const xm::chain::Result &xm::ChainCalibration::result() const {
    return results;
}

void xm::ChainCalibration::debug(bool _debug) {
    DEBUG = _debug;
}
