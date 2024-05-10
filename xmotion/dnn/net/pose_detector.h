//
// Created by henryco on 1/12/24.
//

#ifndef STEREOX_POSE_DETECTOR_H
#define STEREOX_POSE_DETECTOR_H

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include <opencv2/core/mat.hpp>

#include "ssd_anchors.h"
#include "pose_roi.h"
#include "dnn_common.h"
#include "dnn_runner.h"

namespace eox::dnn {

    namespace box {

        enum Model {
            ORIGIN = 0,
            F_32 = 1,
            F_16 = 2
        };

        using Metadata = struct {
            int i_w;
            int i_h;
            int bboxes;
            int box_loc;
            int scores;
            int score_loc;
        };

        const Metadata mappings[3] = {
                {224, 224, 2254, 2254, 0, 1},
                {224, 224, 2254, 2254, 0, 1},
                {128, 128, 896,  896,  1, 0}
        };

        const std::string models[3] = {
                "detection_o_f32.tflite",
                "detection_q_f32.tflite",
                "detection_q_f16.tflite",
        };
    }

    using DetectedPose = struct {

        /**
         * RoI for face
         */
        Box face;

        /**
         * RoI for body
         */
        RoI body;

        /**
         * Key point 0 - mid hip center
         * Key point 1 - point that encodes size & rotation (for full body)
         * Key point 2 - mid shoulder center
         * Key point 3 - point that encodes size & rotation (for upper body)
         */
        Point points[4];

        /**
         * Probability [0,1]
         */
        float score;

        /**
         * from -Pi to Pi radians
         */
        float rotation;
    };

    class PoseDetector : DnnRunner<std::vector<eox::dnn::DetectedPose>> {
        static inline const auto log =
                spdlog::stdout_color_mt("pose_detector");

    private:
        eox::dnn::box::Model model_type = box::ORIGIN;
        eox::dnn::PoseRoi roiPredictor;
        std::vector<std::array<float, 4>> anchors_vec;
        float threshold = 0.5;
        int view_w = 0;
        int view_h = 0;

    protected:
        void initialize() override;

    public:
        std::string get_model_file() override;

        std::vector<DetectedPose> inference(const float *frame) override;

        std::vector<DetectedPose> inference(cv::InputArray &frame);

        void setThreshold(float threshold);

        void set_model_type(box::Model type);

        [[nodiscard]] float getThreshold() const;

        [[nodiscard]] box::Model get_model_type() const;

        [[nodiscard]] int get_in_w() const;

        [[nodiscard]] int get_in_h() const;

        [[nodiscard]] int get_n_bboxes() const;

        [[nodiscard]] int get_n_scores() const;
    };

} // eox

#endif //STEREOX_POSE_DETECTOR_H
