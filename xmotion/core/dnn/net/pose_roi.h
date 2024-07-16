//
// Created by henryco on 1/7/24.
//

#ifndef STEREOX_POSE_ROI_H
#define STEREOX_POSE_ROI_H


#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "dnn_common.h"

namespace eox::dnn {

    using PoseRoiInput = struct {
        eox::dnn::Landmark landmarks[39];
    };

    eox::dnn::Landmark mid(const eox::dnn::PoseRoiInput &input);

    eox::dnn::Landmark end(const eox::dnn::PoseRoiInput &input);

    PoseRoiInput roiFromPoseLandmarks39(const Landmark landmarks[39]);

    PoseRoiInput roiFromPoints(const float mid[2], const float end[2]);

    class PoseRoi {

        static inline const auto log =
                spdlog::stdout_color_mt("pose_roi_predictor");

    private:
        float scale_x = 1;
        float scale_y = 1;
        float fix_x = 0;
        float fix_y = 0;
        float margin = 0;

    public:
        RoI forward(void *data);

        [[nodiscard]] RoI forward(const PoseRoiInput &data) const;

        [[nodiscard]] float getMargin() const;

        [[nodiscard]] float getFixX() const;

        [[nodiscard]] float getFixY() const;

        [[nodiscard]] float getScaleX() const;

        [[nodiscard]] float getScaleY() const;

        PoseRoi &setMargin(float margin);

        PoseRoi &setFixX(float fixX);

        PoseRoi &setFixY(float fixY);

        PoseRoi &setScaleX(float x);

        PoseRoi &setScaleY(float y);

        PoseRoi &setScale(float s);
    };

} // eox

#endif //STEREOX_POSE_ROI_H
