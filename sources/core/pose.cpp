//
// Created by henryco on 4/22/24.
//

#include <opencv2/calib3d.hpp>
#include "../../xmotion/core/algo/pose.h"
#include "../../xmotion/core/utils/cv_utils.h"

void xm::Pose::init(const xm::nview::Initial &params) {
    results.error = false;
    config = params;

    const auto size     = params.devices.size();
    auto       conf_arr = new eox::dnn::pose::PoseInput[size];

    for (int i = 0; i < size; i++) {
        const auto &dev = params.devices[i];
        conf_arr[i]     = {
            .roi_margin = dev.roi_margin,
            .roi_padding_x = dev.roi_padding_x,
            .roi_padding_y = dev.roi_padding_y,
            .roi_scale = dev.roi_scale,
            .roi_rollback_window = dev.roi_rollback_window,
            .roi_center_window = dev.roi_center_window,
            .roi_clamp_window = dev.roi_clamp_window,
            .threshold_marks = dev.threshold_marks,
            .threshold_detector = dev.threshold_detector,
            .threshold_pose = dev.threshold_pose,
            .threshold_roi = dev.threshold_roi,
            .f_v_scale = dev.filter_velocity_factor,
            .f_win_size = dev.filter_windows_size,
            .f_fps = dev.filter_target_fps,
            .bgs_enable = false,
            .bgs_config = {}
        };
    }

    pose_agnostic.init((int) size, conf_arr);

    delete[] conf_arr;

    init_validate();
    init_undistort_maps();
}

void xm::Pose::init_validate() {
//    if (config.epi_matrix.empty())
//        throw std::runtime_error("Epipolar matrix cannot be empty");
    if (config.devices.empty())
        throw std::runtime_error("Devices vector cannot be empty");
//    for (const auto &device: config.devices)
//        if (device.K.empty())
//            throw std::runtime_error("Calibration matrix K for device cannot be empty");
}

void xm::Pose::init_undistort_maps() {
    remap_maps.clear();
    remap_maps.reserve(config.devices.size());
    for (const auto &device: config.devices) {
        if (!device.undistort_source)
            continue;
        auto im_size = cv::Size(device.width, device.height);
        auto new_mat = cv::getOptimalNewCameraMatrix(
                device.K,
                device.D,
                im_size,
                device.undistort_alpha);
        cv::Mat map_1, map_2;
        cv::initUndistortRectifyMap(
                device.K,
                device.D,
                cv::Mat(),
                new_mat,
                im_size,
                CV_16SC2,
                map_1,
                map_2);
        remap_maps.emplace_back(
                new_mat,
                map_1,
                map_2);
    }
}

xm::Pose &xm::Pose::proceed(float delta, const std::vector<xm::ocl::Image2D> &_frames) {
    if (!is_active() || _frames.empty()) {
        images.clear();
        images.reserve(_frames.size());

        results.ready = false;

        if (_frames.size() != config.devices.size()) {
            results.err_msg = "Number of devices != number of frames";
            results.error = true;
            return *this;
        }

        for (auto &img: _frames)
            images.push_back(img);
        return *this;
    }

    const auto n            = _frames.size();
    auto       pose_results = new eox::dnn::pose::PoseResult[n];
    auto       pose_debugs  = new eox::dnn::pose::PoseDebug[n];
    long       duration     = 0;

    pose_agnostic.pass(_frames.data(), pose_results, pose_debugs, duration);

    // TODO

//    for (int i = 0; i < output_frames.size(); i++) {
//        std::vector<std::vector<cv::Vec4f>> epi_vec;
//        const auto pose_output = outputs.at(i);
//        if (!pose_output.present)
//            continue;
//        const auto points = undistorted(pose_output.landmarks, 39, i);
//        const auto mid = points.at(eox::dnn::LM::R_MID);
//        const auto end = points.at(eox::dnn::LM::R_END);
//        for (int j = 0; j < output_frames.size(); j++) {
//            if (i == j)
//                // pointless, so skip
//                continue;
//            const auto mid_line = epi_line_from_point(mid, i, j);
//            const auto end_line = epi_line_from_point(end, i, j);
//            cv::Point2i mp1, mp2, ep1, ep2;
//            points_from_epi_line(output_frames.at(j), mid_line, mp1, mp2);
//            points_from_epi_line(output_frames.at(j), end_line, ep1, ep2);
//            if (DEBUG && config.show_epilines) {
//                const auto color = xm::ocv::distinct_color(i, (int) output_frames.size());
//                cv::line(output_frames.at(j), mp1, mp2, color, 3);
//                cv::line(output_frames.at(j), ep1, ep2, color, 3);
//            }
//        }
//        if (epi_vec.empty())
//            continue;
//    }

// TODO: PROCESS RESULTS ===========================================================================================

    images.clear();
    for (auto &img: _frames)
        images.push_back(img);

    results.ready    = true;
    results.error    = false;
    results.duration = duration;

    return *this;
}

cv::UMat xm::Pose::undistorted(const cv::UMat &in, int index) const {
    if (!config.devices.at(index).undistort_source)
        return in;
    const auto &maps = remap_maps.at(index);
    cv::UMat undistorted;
    cv::remap(in, undistorted, maps.map1, maps.map2, cv::INTER_LINEAR);
    return std::move(undistorted);
}

std::vector<cv::Point2f> xm::Pose::undistorted(const eox::dnn::Landmark *in, int num, int index) const {
    std::vector<cv::Point2f> distorted_points;
    distorted_points.reserve(num);
    for (int i = 0; i < num; ++i)
        distorted_points.emplace_back(in[i].x, in[i].y);

    if (!config.devices.at(index).undistort_points)
        return distorted_points;

    if (config.devices.at(index).undistort_source)
        return distorted_points;

    const auto R = cv::Mat::eye(3, 3, CV_64F);
    const auto K = config.devices.at(index).K.clone();
    const auto D = config.devices.at(index).D.clone();

    std::vector<cv::Point2f> points;
    cv::undistortPoints(distorted_points, points, K, D, R, K);
    return points;
}

void xm::Pose::points_from_epi_line(const cv::UMat &img, const cv::Vec3f &line, cv::Point2i &p1, cv::Point2i &p2) const {
    const float a = line[0];
    const float b = line[1];
    const float c = line[2];

    // Find two points on the line
    if (b != 0) {
        p1.x = 0;
        p1.y = (int) (-c / b);
        p2.x = img.cols;
        p2.y = (int) (-(c + a * (float) img.cols) / b);
    } else {
        p1.x = (int) (-c / a);
        p1.y = 0;
        p2.x = (int) (-(c + b * (float) img.rows));
        p2.y = img.rows;
    }
}

cv::Vec3f xm::Pose::epi_line_from_point(const cv::Point2f &point, int idx_point, int idx_line) const {

    // map point to line: idx_point -> idx_line
    const auto &F = config.epi_matrix[idx_point][idx_line].F;

    {
        const cv::Mat pt = (cv::Mat_<double>(3, 1) << point.x, point.y, 1.f);
        const cv::Mat line = F * pt;
        cv::Vec3f res = {
                (float) line.at<double>(0, 0),
                (float) line.at<double>(1, 0),
                (float) line.at<double>(2, 0)
        };
        const auto norm = (float) std::sqrt(std::pow(res[0], 2) + std::pow(res[1], 2));
        if (norm != 0) {
            res[0] /= norm;
            res[1] /= norm;
            res[2] /= norm;
        }
        return res;
    }
}

