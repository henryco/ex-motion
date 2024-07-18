//
// Created by henryco on 16/07/24.
//

#include "../../xmotion/core/pose/roi/roi_body_heuristics.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "UnusedValue"
namespace xm::pose::roi {

    void RoiBodyHeuristics::reset() {
        prediction = false;
        roi = {};
    }

    void RoiBodyHeuristics::init() {
        initialized = false;
        reset();
    }

    float RoiBodyHeuristics::presence(const eox::dnn::RoI &_roi, const eox::dnn::Landmark *landmarks, bool auto_reset) {
        const auto score = presence(_roi, landmarks);
        if (score < 0 && auto_reset)
            reset();
        return score;
    }

    float RoiBodyHeuristics::presence(const eox::dnn::RoI &_roi, const eox::dnn::Landmark *landmarks) const {
        float roi_score = 0;

        // ROI threshold check when its reasonable
        if (threshold > 0 && threshold < 1) {
            // points of interest for ROI threshold check
            constexpr int POI[] = {
                    eox::dnn::LM::NOSE,
                    eox::dnn::LM::R_MID,
                    eox::dnn::LM::HIP_L,
                    eox::dnn::LM::HIP_R,
                    eox::dnn::LM::SHOULDER_L,
                    eox::dnn::LM::SHOULDER_R,
            };

            // ROI threshold check, currently only horizontal
            for (const auto &i: POI) {
                const auto &point = landmarks[i];

                // ROI radius (actually half of the horizontal length)
                const auto roi_r = _roi.w * .5f;

                // ROI center (actually horizontal part of the center)
                const auto roi_c = _roi.x + roi_r;

                // normalized horizontal position of the point
                const auto position = std::abs(point.x - roi_c) / roi_r;

                // normalized distance of point from the ROI border
                const auto distance = 1.f - position;

                // for debug purpose
                roi_score = roi_score > 0 ? std::min(distance, roi_score) : distance;

                // too close to the ROI borders
                if (distance < threshold)
                    return -roi_score;
            }
        }

        return roi_score;
    }

    RoiBodyHeuristicsResult RoiBodyHeuristics::predict(
            const eox::dnn::RoI &_roi,
            const eox::dnn::Landmark landmarks[39],
            int width,
            int height) {
        const bool first_run = !initialized;

        bool preserved_roi = false;
        bool discarded_roi = false;
        bool rollback_roi = false;

        if (!initialized) {
            initialized = true;
        }

        const auto previous_roi = roi;
        roi = _roi;

        // detector run due to roi being discarded previously
        if (!prediction && !first_run && rollback_window > 0) {

            // MID point extracted from landmarks
            const auto found_mid = landmarks[eox::dnn::LM::R_MID];

            // MID point extracted from detector
            const auto expected_mid = roi.c;

            // radius of the expected roi circle
            const auto roi_radius = std::sqrt(
                    std::pow(roi.c.x - roi.e.x, 2.f)
                    + std::pow(roi.c.y - roi.e.y, 2.f));

            // Euclidean distance between expected and expected mid-points
            const auto dist = std::sqrt(
                    std::pow(found_mid.x - expected_mid.x, 2.f)
                    + std::pow(found_mid.y - expected_mid.y, 2.f));

            // normalized distance
            const auto norm_dist = dist / roi_radius;

            // if far enough from expected mid, rolling back to pre-discarded roi
            roi = (norm_dist < rollback_window)
                  ? roi
                  : previous_roi;

            rollback_roi = norm_dist >= rollback_window;
        }

        // saving old roi just in case
        const auto roi_old = roi;

        // predict new roi
        roi = to_roi(landmarks);

        // checking if we really need to use new roi
        if (prediction && center_window > 0) {

            // Euclidean distance between mid-points of new and old roi
            const float c_dist = std::sqrt(
                    std::pow(roi.c.x - roi_old.c.x, 2.f)
                    + std::pow(roi.c.y - roi_old.c.y, 2.f));

            // radius of the old roi circle
            const float radius = std::sqrt(
                    std::pow(roi_old.c.x - roi_old.e.x, 2.f)
                    + std::pow(roi_old.c.y - roi_old.e.y, 2.f));

            // normalized distance
            const float norm_dist = c_dist / radius;

            roi = (norm_dist >= center_window)
                  ? roi
                  : roi_old;

            preserved_roi = norm_dist < center_window;
        }

        // clamping roi to prevent index out of range error
        const auto clamped_roi = eox::dnn::clamp_roi(roi, width, height);
        const auto ratio_roi_w = clamped_roi.w / roi.w;
        const auto ratio_roi_h = clamped_roi.h / roi.h;

        // new roi is clamped roi
        roi = clamped_roi;

        // checking if clamped roi big enough
        if (ratio_roi_w > clamp_window && ratio_roi_h > clamp_window) {
            // it is, we can use it
            discarded_roi = false;
            prediction = true;
        } else {
            // it's not, gotta use detector
            discarded_roi = true;
            prediction = false;
        }

        return {
                .roi = roi,
                .preserved = preserved_roi,
                .discarded = discarded_roi,
                .rollback = rollback_roi
        };
    }

    eox::dnn::RoI RoiBodyHeuristics::to_roi(const eox::dnn::Landmark *landmarks) const {
        return eox::dnn::to_roi(
                landmarks[eox::dnn::LM::R_MID],
                landmarks[eox::dnn::LM::R_END],
                margin,
                scale,
                padding_x,
                padding_y
        );
    }

    const bool RoiBodyHeuristics::get_prediction() const {
        return prediction;
    }

}


#pragma clang diagnostic pop