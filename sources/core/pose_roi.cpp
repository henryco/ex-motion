//
// Created by henryco on 1/7/24.
//

#include "../../xmotion/core/dnn/net/pose_roi.h"

#include <cmath>

namespace eox::dnn {

    eox::dnn::Landmark mid(const eox::dnn::PoseRoiInput &input) {
        return input.landmarks[eox::dnn::LM::R_MID];
    }

    eox::dnn::Landmark end(const eox::dnn::PoseRoiInput &input) {
        return input.landmarks[eox::dnn::LM::R_END];
    }

    PoseRoiInput roiFromPoseLandmarks39(const Landmark landmarks[39]) {
        PoseRoiInput output;
        for (int i = 0; i < 39; i++)
            output.landmarks[i] = landmarks[i];
        return output;
    }

    PoseRoiInput roiFromPoints(const float mid[2], const float end[2]) {
        PoseRoiInput output;
        output.landmarks[eox::dnn::LM::R_MID] = {.x = mid[0], .y = mid[1], .v = 1, .p = 1};
        output.landmarks[eox::dnn::LM::R_END] = {.x = end[0], .y = end[1], .v = 1, .p = 1};
        return output;
    }

    RoI PoseRoi::forward(void *data) {
        return forward(*static_cast<eox::dnn::PoseRoiInput*>(data));
    }

    RoI PoseRoi::forward(const eox::dnn::PoseRoiInput &data) const {
        const auto &center = data.landmarks[eox::dnn::LM::R_MID];
        const auto &end = data.landmarks[eox::dnn::LM::R_END];

        const float x1 = center.x;
        const float y1 = center.y;
        const float x2 = end.x;
        const float y2 = end.y;

        const float dist = std::sqrt(std::pow(x2 - x1, 2.f) + std::pow(y2 - y1, 2.f));
        const float radius = dist + margin;

        const float w = radius * 2.f * scale_x;
        const float h = radius * 2.f * scale_y;

        const float x0 = x1 - (w / 2.f);
        const float y0 = y1 - (h / 2.f);

        const float ex = x1 + (((x2 - x1) / dist) * radius * scale_x);
        const float ey = y1 + (((y2 - y1) / dist) * radius * scale_y);

        return {
                .x = std::max(0.f, x0 + fix_x),
                .y = std::max(0.f, y0 + fix_y),
                .w = std::max(0.f, w),
                .h = std::max(0.f, h),
                .c = Point(x1, y1),
                .e = Point(ex, ey)
        };
    }

    float PoseRoi::getFixY() const {
        return fix_y;
    }

    float PoseRoi::getMargin() const {
        return margin;
    }

    PoseRoi &PoseRoi::setFixX(float fixX) {
        fix_x = fixX;
        return *this;
    }

    PoseRoi &PoseRoi::setFixY(float fixY) {
        fix_y = fixY;
        return *this;
    }

    PoseRoi &PoseRoi::setMargin(float _margin) {
        margin = _margin;
        return *this;
    }

    PoseRoi &PoseRoi::setScaleX(float x) {
        scale_x = x;
        return *this;
    }

    PoseRoi &PoseRoi::setScaleY(float y) {
        scale_y = y;
        return *this;
    }

    PoseRoi &PoseRoi::setScale(float s) {
        scale_x = s;
        scale_y = s;
        return *this;
    }

    float PoseRoi::getFixX() const {
        return fix_x;
    }

    float PoseRoi::getScaleX() const {
        return scale_x;
    }

    float PoseRoi::getScaleY() const {
        return scale_y;
    }

} // eox