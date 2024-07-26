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

}