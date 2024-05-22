//
// Created by henryco on 4/24/24.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "../../xmotion/core/utils/cv_utils.h"

namespace xm::ocv {

    void clamp(cv::InputOutputArray &mat, double min, double max) {
        cv::threshold(mat, mat, min, min, cv::THRESH_TOZERO);
        cv::threshold(mat, mat, max, max, cv::THRESH_TRUNC);
    }

    cv::Mat img_copy(const cv::Mat &image, int color_space_conv_type, int matrix_data_type) {
        cv::Mat output;
        image.convertTo(output, matrix_data_type);
        return img_copy(output, color_space_conv_type);
    }

    cv::Mat img_copy(const cv::Mat &image, int color_space_conv_type) {
        cv::Mat output;
        cv::cvtColor(image, output, color_space_conv_type);
        return std::move(output);
    }

    cv::Mat img_copy(const cv::Mat &image) {
        cv::Mat output;
        image.copyTo(output);
        return std::move(output);
    }

    Squares find_squares(const cv::Mat &image, uint columns, uint rows, bool sb, int flag) {
        cv::Mat gray = img_copy(image, cv::COLOR_BGR2GRAY);
        cv::Mat copy = img_copy(image);

        const auto size = cv::Size((int) columns - 1, (int) rows - 1);

        bool found;
        std::vector<cv::Point2f> corners;

        if (!sb) {
            const int flags = flag
                              | cv::CALIB_CB_NORMALIZE_IMAGE
                              | cv::CALIB_CB_FILTER_QUADS
                              | cv::CALIB_CB_ADAPTIVE_THRESH
                              | cv::CALIB_CB_ACCURACY
                              | cv::CALIB_CB_FAST_CHECK
                              | cv::CALIB_CB_EXHAUSTIVE;
            found = cv::findChessboardCorners(gray, size, corners, flags);
        }

        else {
            const int flags = flag
                              | cv::CALIB_CB_NORMALIZE_IMAGE
                              | cv::CALIB_CB_ACCURACY
                              | cv::CALIB_CB_MARKER
                              | cv::CALIB_CB_EXHAUSTIVE;
            found = cv::findChessboardCornersSB(gray, size, corners, flags);
        }

        if (found && !sb) {
            const auto term = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 60, 0.001);
            const auto window = cv::Size(11, 11);
            const auto zone = cv::Size(-1, -1);
            cv::cornerSubPix(gray, corners, window, zone, term);
        }

        cv::drawChessboardCorners(copy, size, corners, found);
        return {
                .corners = std::move(corners),
                .original = image,
                .result = std::move(copy),
                .found = found
        };
    }

    cv::Scalar distinct_color(int index, int N) {
        // Ensure the index is within bounds
        if (index < 0 || index >= N)
            throw std::out_of_range("Index out of bounds");

        // Calculate hue value in degrees (0-360) equally spaced for N colors
        // Convert hue to the range [0, 179] as used in OpenCV
        const auto hue = (360.0 * index / N) / 2;

        // HSV values (hue, saturation, value)
        const cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));

        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        // Extract BGR values
        const cv::Vec3b bgrPixel = bgr.at<cv::Vec3b>(0, 0);
        return cv::Scalar(bgrPixel[0], bgrPixel[1], bgrPixel[2]); // NOLINT(*-return-braced-init-list)
    }

}
