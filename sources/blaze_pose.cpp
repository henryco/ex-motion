//
// Created by henryco on 12/29/23.
//

#include "../xmotion/dnn/net/blaze_pose.h"
#include <filesystem>

namespace eox::dnn {

    const float *lm_3d_1x195(const tflite::Interpreter &interpreter, pose::Model model) {
        return interpreter.output_tensor(pose::mappings[model].lm_3d)->data.f;
    }

    const float *lm_world_1x117(const tflite::Interpreter &interpreter, pose::Model model) {
        return interpreter.output_tensor(pose::mappings[model].world)->data.f;
    }

    const float *heatmap_1x64x64x39(const tflite::Interpreter &interpreter, pose::Model model) {
        return interpreter.output_tensor(pose::mappings[model].hm)->data.f;
    }

    const float *segmentation_1x128x128x1(const tflite::Interpreter &interpreter, pose::Model model) {
        return interpreter.output_tensor(pose::mappings[model].seg)->data.f;
    }

    const float *pose_flag_1x1(const tflite::Interpreter &interpreter, pose::Model model) {
        return interpreter.output_tensor(pose::mappings[model].flag)->data.f;
    }

//    const std::vector<std::string> BlazePose::outputs = {
//            "Identity:0",   // 598 | 0: [1, 195]           landmarks 3d
//            "Identity_4:0", // 600 | 1: [1, 117]           world 3d
//            "Identity_3:0", // 481 | 2: [1, 64, 64, 39]    heatmap
//            "Identity_2:0", // 608 | 3: [1, 128, 128, 1]   segmentation
//            "Identity_1:0", // 603 | 4: [1, 1]             pose flag (score)
//    };

    PoseOutput BlazePose::inference(cv::InputArray &frame) {
        auto ref = frame.getMat();
        view_w = ref.cols;
        view_h = ref.rows;

        // [1, 3, 256, 256] or [1, 3, 128, 128]
        cv::Mat blob = eox::dnn::convert_to_squared_blob(ref, get_in_w(), get_in_h(), true);
        return inference(blob.ptr<float>(0));
    }

    PoseOutput BlazePose::inference(const float *frame) {
        init();

        input(0, frame, get_in_w() * get_in_h() * 3 * 4);
        invoke();

        PoseOutput output;

        const auto presence = *pose_flag_1x1(*interpreter, model_type);
        output.score = presence;

        const float *land_marks_3d = lm_3d_1x195(*interpreter, model_type);
        const float *land_marks_wd = lm_world_1x117(*interpreter, model_type);

        // correcting letterbox paddings
        const auto p = eox::dnn::get_letterbox_paddings(view_w, view_h, get_in_w(), get_in_h());
        const auto n_w = (float) get_in_w() - (p.left + p.right);
        const auto n_h = (float) get_in_h() - (p.top + p.bottom);

        for (int i = 0; i < 39; i++) {
            const int j = i * 3;
            const int k = i * 5;
            // normalized landmarks_3d
            output.landmarks_norm[i] = {
                    .x = (land_marks_3d[k + 0] - p.left) / n_w,
                    .y = (land_marks_3d[k + 1] - p.top) / n_h,
                    .z = land_marks_3d[k + 2] / (float) std::max(get_in_w(), get_in_h()),
                    .v = land_marks_3d[k + 3],
                    .p = land_marks_3d[k + 4],
            };

            // world-space landmarks
            output.landmarks_3d[i] = {
                    .x = land_marks_wd[j + 0],
                    .y = land_marks_wd[j + 1],
                    .z = land_marks_wd[j + 2],
            };
        }

        if (SEGMENTATION) {
            const float *s = segmentation_1x128x128x1(*interpreter, model_type);

            // 256x256 or 128x128
            const auto size = get_in_w() * get_in_h();

            if (size == 256 * 256) {

                for (int i = 0; i < size; i++)
                    output.segmentation[i] = (float) eox::dnn::sigmoid(s[i]);

            } else {

                auto *buffer = new float[size];
                for (int i = 0; i < size; i++)
                    buffer[i] = (float) eox::dnn::sigmoid(s[i]);

                cv::Mat temp = cv::Mat(get_in_h(), get_in_w(), CV_32F, buffer);
                cv::Mat dst;
                cv::resize(temp, dst, cv::Size(256, 256), 0, 0, cv::INTER_NEAREST);

                if (dst.isContinuous()) {
                    memcpy(output.segmentation, dst.data, 256 * 256 * sizeof(float));
                }

                delete[] buffer;
            }
        }

        return output;
    }

    std::string BlazePose::get_model_file() {
        return "./../models/blazepose/body/whole/" + pose::models[model_type];
    }

    bool BlazePose::segmentation() const {
        return SEGMENTATION;
    }

    void BlazePose::set_segmentation(bool segmentation) {
        SEGMENTATION = segmentation;
    }

    pose::Model BlazePose::get_model_type() const {
        return model_type;
    }

    void BlazePose::set_model_type(pose::Model type) {
        model_type = type;
    }

    int BlazePose::get_in_w() const {
        return pose::mappings[model_type].i_w;
    }

    int BlazePose::get_in_h() const {
        return pose::mappings[model_type].i_h;
    }

} // eox