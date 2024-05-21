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
        if (index < 0 || index >= N) {
            throw std::out_of_range("Index out of range");
        }

        // Convert index to hue in HSV color space
        int hue = (index * 360 / N) % 360;  // Hue varies from 0 to 360

        // Saturation and value are fixed to max for bright colors
        int saturation = 255;
        int value = 255;

        // Create HSV color
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, saturation, value));

        // Convert HSV color to BGR color
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        // Return the BGR color as cv::Scalar
        cv::Vec3b bgrVec = bgr.at<cv::Vec3b>(0, 0);
        return cv::Scalar(bgrVec[0], bgrVec[1], bgrVec[2]); // NOLINT(*-return-braced-init-list)
    }

    cv::Mat inverse(const cv::Mat &in) {
        cv::Mat out;
        cv::invert(in, out);
        return out;
    }

    std::string print_matrix(const cv::Mat &in) {
        const int w = in.cols;
        const int h = in.rows;
        std::string result = "\n";
        result += "[\n";
        for (int y = 0; y < h; y++) {
            result + "[";
            for (int x = 0; x < w; x++) {
                if (x != 0)
                    result += ", ";
                result += std::to_string(in.at<float>(x, y));
            }
            result + "]\n";
        }

        return result + "]";
    }


}
