//
// Created by henryco on 4/22/24.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
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

void xm::Calibration::calibrate() {
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

    // output parameters
    cv::Mat camera_matrix, distortion_coefficients;
    std::vector<cv::Mat> r_vecs, t_vecs;
    std::vector<double> std_intrinsics, std_extrinsics, per_view_errors;

    int flags = 0;
    const cv::TermCriteria criteria(
            cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
            60,
            DBL_EPSILON);

    if ((config.fx != -1 && config.fy != -1) ||
        (config.cx != -1 && config.cy != -1)) {

        flags |= cv::CALIB_USE_INTRINSIC_GUESS;
        flags |= (config.fix_f ? cv::CALIB_FIX_FOCAL_LENGTH : 0);
        flags |= (config.fix_c ? cv::CALIB_FIX_PRINCIPAL_POINT : 0);

        camera_matrix = cv::Mat::zeros(3, 3, CV_32F);

        camera_matrix.at<float>(0, 0) = config.fx;
        camera_matrix.at<float>(1, 1) = config.fy;
        camera_matrix.at<float>(0, 2) = config.cx;
        camera_matrix.at<float>(1, 2) = config.cy;
        camera_matrix.at<float>(2, 2) = 1.f;

        if (config.cx == -1 || config.cy == -1) {
            camera_matrix.at<float>(0, 2) = ((float) config.width) / 2.f;
            camera_matrix.at<float>(1, 2) = ((float) config.height) / 2.f;
        }

        if (config.fx == -1 || config.fy == -1) {
            throw std::runtime_error("F_x and F_y should be specified");
        }
    }

    // calibration
    const auto rms = cv::calibrateCamera(
            object_points,
            image_points,
            cv::Size((int) config.width, (int) config.height),
            camera_matrix,
            distortion_coefficients,
            r_vecs,
            t_vecs,
            std_intrinsics,
            std_extrinsics,
            per_view_errors,
            flags,
            criteria
    );

    // calculating mean re-projection error
    double totalError = 0;
    size_t totalPoints = 0;
    std::vector<cv::Point2f> reprojectedPoints;
    for (size_t i = 0; i < object_points.size(); ++i) {
        cv::projectPoints(
                object_points[i],
                r_vecs[i],
                t_vecs[i],
                camera_matrix,
                distortion_coefficients,
                reprojectedPoints
        );
        size_t n = object_points[i].size();
        for (size_t j = 0; j < n; ++j) {
            double err = norm(image_points[i][j] - reprojectedPoints[j]);
            totalError += err * err;
        }
        totalPoints += n;
    }

    const double mre = sqrt(totalError / (double) totalPoints);

    results.K = camera_matrix;
    results.D = distortion_coefficients;
    results.mre_1 = rms;
    results.mre_2 = mre;
    results.ready = true;
    results.remains_ms = 0;
    results.remains_cap = 0;
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

    calibrate();

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
    results.ready = false;

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