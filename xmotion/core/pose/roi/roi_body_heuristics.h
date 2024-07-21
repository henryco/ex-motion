//
// Created by henryco on 16/07/24.
//

#ifndef XMOTION_ROI_BODY_HEURISTICS_H
#define XMOTION_ROI_BODY_HEURISTICS_H

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "../../dnn/net/dnn_common.h"
#include "../../ocl/ocl_data.h"

namespace xm::pose::roi {

    typedef struct {
        eox::dnn::RoI roi;
        bool prediction;
        bool preserved;
        bool discarded;
        bool rollback;
    } RoiBodyHeuristicsResult;

    class RoiBodyHeuristics {

        static inline const auto log =
                spdlog::stdout_color_mt("roi_body_predictor");

    private:
        bool initialized = false;
        bool prediction = false;

        eox::dnn::RoI roi{};

    public:

        /**
         * Margins added to ROI
         */
        float margin = 0.f;

        /**
         * Horizontal paddings added to ROI
         */
        float padding_x = 0.f;

        /**
         * Vertical paddings added to ROI
         */
        float padding_y = 0.f;

        /**
         * Scaling factor for ROI (multiplication)
         */
        float scale = 1.2f;

        /**
         * Distance between detectors and actual ROI mid point
         * for which detected ROI should be rolled back to previous one
         *
         * [0.0 ... 1.0]
         */
        float rollback_window = 0.f;

        /**
         * Distance between actual and predicted ROI mid point
         * for which should stay unchanged (helps reducing jittering)
         *
         * [0.0 ... 1.0]
         */
        float center_window = 0.f;

        /**
         * Acceptable ratio of clamped to original ROI size. \n
         * Zero (0) means every size is acceptable \n
         * One (1) means only original (non-clamped) ROI is acceptable \n
         *
         * [0.0 ... 1.0]
         */
        float clamp_window = 0.f;

        /**
         * Threshold score for detector ROI distance to body marks. \n
         * In other works: How far marks should be from detectors ROI borders. \n
         * Currently implemented only for horizontal axis.
         *
         * [0.0 ... 1.0]
         */
        float threshold = 0.f;

    public:

        /**
         * @return whether the prediction from the last pass was successful
         */
        const bool get_prediction() const;

        /**
         * Wipes completely internal state to its initial values.
         */
        void init();

        /**
         * Reset internal state
         */
        void reset();

        /**
         * @param roi tested region of interest
         * @param landmarks detected body landmarks
         * @param auto_reset reset when roi is disqualified
         * @return roi presence score, negative when roi is disqualified
         */
        float presence(
                const eox::dnn::RoI &roi,
                const eox::dnn::Landmark landmarks[39],
                bool auto_reset);

        /**
         * @param roi tested region of interest
         * @param landmarks detected body landmarks
         * @return roi presence score, negative when roi is disqualified
         */
        float presence(const eox::dnn::RoI &roi,
                       const eox::dnn::Landmark landmarks[39]) const;

        /**
         * Predict new roi, based on some heuristics
         * @param landmarks detected body landmarks
         * @param width of the analyzed original image
         * @param height of the analyzed original image
         * @return predicted roi with extra metadata
         */
        RoiBodyHeuristicsResult predict(
                const eox::dnn::RoI &roi,
                const eox::dnn::Landmark landmarks[39],
                int width,
                int height);

        /**
         * @param landmarks detected body landmarks
         * @return new roi extracted from landmarks
         */
        eox::dnn::RoI to_roi(const eox::dnn::Landmark landmarks[39]) const;
    };
}

#endif //XMOTION_ROI_BODY_HEURISTICS_H
