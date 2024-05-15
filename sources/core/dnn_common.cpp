//
// Created by henryco on 1/5/24.
//

#import "../../xmotion/core/dnn/net/dnn_common.h"

#import <cmath>
#include <opencv2/imgproc.hpp>

namespace eox::dnn {

    /**
     * see media/pose_landmark_topology.svg
     */
    const int body_joints[31][2] = {
            {0,  2},
            {0,  5},
            {2,  7},
            {5,  8},

            {10, 9},

            {12, 11},

            {12, 14},
            {14, 16},
            {16, 22},
            {16, 18},
            {16, 20},
            {18, 20},

            {11, 13},
            {13, 15},
            {15, 21},
            {15, 17},
            {15, 19},
            {17, 19},

            {12, 24},
            {24, 23},
            {11, 23},

            {24, 26},
            {26, 28},
            {28, 32},
            {28, 30},
            {32, 30},

            {23, 25},
            {25, 27},
            {27, 29},
            {27, 31},
            {29, 31},
    };

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    cv::Mat convert_to_squared_blob(const cv::Mat &in, int size, bool keep_aspect_ratio) {
        return convert_to_squared_blob(in, size, size, keep_aspect_ratio);
    }

    cv::Mat convert_to_squared_blob(const cv::Mat &in, int width, int height, bool keep_aspect_ratio) {
        cv::Mat blob;

        if (in.cols != width || in.rows != height) {

            if (keep_aspect_ratio) {
                // letterbox, preserving aspect ratio

                const float scale = std::min((float) width / (float) in.cols, (float) height / (float) in.rows);
                const float n_w = (float) in.cols * scale;
                const float n_h = (float) in.rows * scale;

                const float s_x = ((float) width - n_w) / 2.f;
                const float s_y = ((float) height - n_h) / 2.f;

                blob = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
                cv::Mat roi = blob(cv::Rect((int) s_x, (int) s_y, (int) n_w, (int) n_h));
                cv::resize(in, roi, cv::Size((int) n_w, (int) n_h),
                           0, 0, cv::INTER_CUBIC);
            } else {
                // resize without preserving aspect ratio
                cv::resize(in, blob, cv::Size(width, height),
                           0, 0, cv::INTER_CUBIC);
            }

        } else {
            in.copyTo(blob);
        }

        cv::cvtColor(blob, blob, cv::COLOR_BGR2RGB);
        blob.convertTo(blob, CV_32FC3, 1.0 / 255.);

        return blob;
    }

    Paddings get_letterbox_paddings(int width, int height, int size) {
        return get_letterbox_paddings(width, height, size, size);
    }

    Paddings get_letterbox_paddings(int width, int height, int box_w, int box_h) {
        const float scale = std::min((float) box_w / (float) width, (float) box_h / (float) height);
        const int n_w = (int) ((float) width * scale);
        const int n_h = (int) ((float) height * scale);

        const int s_x = (int) ((float) (box_w - n_w) / 2.f);
        const int s_y = (int) ((float) (box_h - n_h) / 2.f);

        return {
                .left = (float) s_x,
                .right = (float) s_x,
                .top = (float) s_y,
                .bottom = (float) s_y,
        };
    }

    cv::Mat remove_paddings(const cv::Mat &in, int width, int height) {
        const auto paddings = get_letterbox_paddings(
                width, height, in.cols, in.rows
        );
        return in(cv::Rect(
                paddings.left,
                paddings.top,
                width - (paddings.left + paddings.right),
                height - (paddings.top + paddings.bottom)
        ));
    }

    RoI clamp_roi(const RoI &in, int width, int height) {
        auto roi = RoI(in.x, in.y, in.w, in.h, in.c, in.e);

        roi.x = std::clamp(in.x, 0.f, (float) width - 1);
        roi.y = std::clamp(in.y, 0.f, (float) height - 1);
        roi.w = std::clamp(in.w, 0.f, (float) width - roi.x - 1);
        roi.h = std::clamp(in.h, 0.f, (float) height - roi.y - 1);
        roi.c = {
                .x = std::clamp(in.c.x, 0.f, (float) width - 1),
                .y = std::clamp(in.c.y, 0.f, (float) height - 1),
        };
        roi.e = {
                .x = std::clamp(in.e.x, 0.f, (float) width - 1),
                .y = std::clamp(in.e.y, 0.f, (float) height - 1),
        };

        return roi;
    }

    double normalize_radians(double angle) {
        return angle - 2 * M_PI * floor((angle + M_PI) / (2 * M_PI));
    }

}