//
// Created by henryco on 5/10/24.
//

#include "../xmotion/dnn/pose_pipeline.h"

namespace eox::dnn {

    void PosePipeline::drawJoints(const eox::dnn::Landmark landmarks[39], cv::Mat &output) const {
        for (auto bone: eox::dnn::body_joints) {
            const auto &start = landmarks[bone[0]];
            const auto &end = landmarks[bone[1]];
            if (eox::dnn::sigmoid(start.p) > threshold_presence && eox::dnn::sigmoid(end.p) > threshold_presence) {
                cv::Point sp(start.x, start.y);
                cv::Point ep(end.x, end.y);
                cv::Scalar color(230, 0, 230);
                cv::line(output, sp, ep, color, 2);
            }
        }
    }

    void PosePipeline::drawLandmarks(const eox::dnn::Landmark landmarks[39], const eox::dnn::Coord3d ws3d[39], cv::Mat &output) const {
        for (int i = 38; i >= 0; i--) {
            const auto point = landmarks[i];
            const auto visibility = eox::dnn::sigmoid(point.v);
            const auto presence = eox::dnn::sigmoid(point.p);
            if (presence > threshold_presence || i > 32) {
                cv::Point circle(point.x, point.y);

                if (i > 32) {
                    const auto color = cv::Scalar(0, 0, 255);
                    cv::circle(output, circle, 8, color, 2);
                }
                else {
                    const auto color = cv::Scalar(255 * (1.f - visibility), 255 * visibility, 0);
                    cv::circle(output, circle, 2, color, 4);
                }


//                if (i <= 32) {
//                    cv::putText(output, std::to_string(visibility), cv::Point(circle.x - 10, circle.y - 10),
//                                cv::FONT_HERSHEY_SIMPLEX, 0.7,
//                                cv::Scalar(0, 0, 255), 2);
//                }
//
//                if (i == 25) {
//                    cv::putText(output,
//                                std::to_string(point.x / output.cols) + ", " +
//                                std::to_string(point.y / output.rows) + ", " +
//                                std::to_string(point.z),
//                                cv::Point(40, 40),
//                                cv::FONT_HERSHEY_SIMPLEX, 0.7,
//                                cv::Scalar(0, 0, 255), 2);
//
//                    cv::putText(output,
//                                std::to_string(ws3d[i].x) + ", " +
//                                std::to_string(ws3d[i].y) + ", " +
//                                std::to_string(ws3d[i].z),
//                                cv::Point(40, 80),
//                                cv::FONT_HERSHEY_SIMPLEX, 0.7,
//                                cv::Scalar(0, 0, 255), 2);
//                }
            }
        }
    }

    void PosePipeline::drawRoi(cv::Mat &output) const {
        const auto p1 = cv::Point(roi.x, roi.y);
        const auto p2 = cv::Point(roi.x + roi.w, roi.y + roi.h);
        cv::Scalar color(0, 255, 0);
        cv::line(output, p1, cv::Point(roi.x + roi.w, roi.y), color, 2);
        cv::line(output, p1, cv::Point(roi.x, roi.y + roi.h), color, 2);
        cv::line(output, p2, cv::Point(roi.x, roi.y + roi.h), color, 2);
        cv::line(output, p2, cv::Point(roi.x + roi.w, roi.y), color, 2);

        cv::Point mid(roi.c.x, roi.c.y);
        cv::Point end(roi.e.x, roi.e.y);
        cv::Scalar pc(0, 255, 255);
        cv::circle(output, mid, 3, pc, 5);
        cv::circle(output, end, 3, pc, 5);

        cv::putText(output,
                    std::to_string(mid.x) + ", " +std::to_string(mid.y),
                    cv::Point(40, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);

        cv::putText(output,
                    std::to_string(end.x) + ", " +std::to_string(end.y),
                    cv::Point(40, 160),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);
    }

}