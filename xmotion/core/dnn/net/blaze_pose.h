//
// Created by henryco on 12/29/23.
//

#ifndef STEREOX_BLAZE_POSE_H
#define STEREOX_BLAZE_POSE_H

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

#include "dnn_common.h"
#include "dnn_runner.h"

namespace eox::dnn {

    namespace pose {

        enum Model {
            HEAVY_ORIGIN = 0,
            FULL_ORIGIN = 1,
            LITE_ORIGIN = 2,

            HEAVY_F32 = 3,
            FULL_F32 = 4,
            LITE_F32 = 5,

            HEAVY_F16 = 6,
            FULL_F16 = 7,
            LITE_F16 = 8
        };

        using Metadata = struct {
            int i_w;
            int i_h;
            int o_w;
            int o_h;
            int lm_3d;
            int world;
            int hm;
            int seg;
            int flag;
        };

        const Metadata mappings[9] = {
                {256, 256, 256, 256, 0, 4, 3, 2, 1},
                {256, 256, 256, 256, 0, 4, 3, 2, 1},
                {256, 256, 256, 256, 0, 4, 3, 2, 1},

                {256, 256, 128, 128, 0, 1, 2, 3, 4},
                {256, 256, 128, 128, 0, 1, 2, 3, 4},
                {256, 256, 128, 128, 0, 1, 4, 2, 3},

                {256, 256, 128, 128, 0, 1, 2, 3, 4},
                {256, 256, 256, 256, 2, 3, 4, 0, 1},
                {256, 256, 128, 128, 0, 1, 4, 2, 3},
        };

        const std::string models[9] = {
                "landmark_heavy_o_f32.tflite",
                "landmark_full_o_f32.tflite",
                "landmark_lite_o_f32.tflite",

                "landmark_heavy_q_f32.tflite",
                "landmark_full_q_f32.tflite",
                "landmark_lite_q_f32.tflite",

                "landmark_heavy_q_f16.tflite",
                "landmark_full_q_f16.tflite",
                "landmark_lite_q_f16.tflite",
        };
    }

    class BlazePose : DnnRunner<PoseOutput> {
        static inline const auto log =
                spdlog::stdout_color_mt("blaze_pose");

    private:
        eox::dnn::pose::Model model_type = pose::HEAVY_ORIGIN;
        bool with_box = false;
        int view_w = 0;
        int view_h = 0;

    protected:
        std::string get_model_file() override;

        PoseOutput inference() override;

    public:
        bool SEGMENTATION = true;

        /**
         * @param frame BGR image (ie. cv::Mat of CV_8UC3)
         */
        PoseOutput inference(const cv::Mat &frame);

        /**
         * @param frame BGR image (ie. cv::Mat of CV_8UC3)
         */
        PoseOutput inference(const cv::UMat &frame);

        /**
         * @param frame pointer to 256x256 row-oriented 1D array representation of 256x256x3 RGB image
         */
        PoseOutput inference(const float *frame);

        void set_segmentation(bool segmentation);

        void set_model_type(pose::Model type);

        [[nodiscard]] pose::Model get_model_type() const;

        [[nodiscard]] bool segmentation() const;

        [[nodiscard]] int get_in_w() const;

        [[nodiscard]] int get_in_h() const;
    };

} // eox

#endif //STEREOX_BLAZE_POSE_H
