//
// Created by henryco on 19/07/24.
//

#include "../agnostic_body.h"

namespace platform::dnn {

    namespace body {
        const Metadata mappings[9] = {
            {256, 256, 256, 256, 0, 2, 1},
            {256, 256, 256, 256, 0,  2, 1},
            {256, 256, 256, 256, 0,  2, 1},

            {256, 256, 128, 128, 0,  3, 4},
            {256, 256, 128, 128, 0,  3, 4},
            {256, 256, 128, 128, 0,  2, 3},

            {256, 256, 128, 128, 0,  3, 4},
            {256, 256, 256, 256, 2,  0, 1},
            {256, 256, 128, 128, 0,  2, 3},
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

    AgnosticBody::AgnosticBody(body::Model model) {
        // TODO
    }

    AgnosticBody::~AgnosticBody() {
        // TODO
    }

    body::Model AgnosticBody::get_model() const {
        return body::Model::HEAVY_F32; // TODO
    }

    size_t AgnosticBody::get_in_w() const {
        return 0; // TODO
    }

    size_t AgnosticBody::get_in_h() const {
        return 0; // TODO
    }

    size_t AgnosticBody::get_n_lm3d() const {
        return 0; // TODO
    }

    size_t AgnosticBody::get_n_segmentation_w() const {
        return 0; // TODO
    }

    size_t AgnosticBody::get_n_segmentation_h() const {
        return 0; // TOOD
    }

    void AgnosticBody::inference(size_t batch_size, const float *in_batch_ptr) {
        // TODO
    }

    const float *const *AgnosticBody::get_segmentations() const {
        return nullptr; // TODO
    }

    const float *const *AgnosticBody::get_landmarks_3d() const {
        return nullptr; // TODO
    }

    const float *const *AgnosticBody::get_pose_flags() const {
        return nullptr; // TODO
    }

}