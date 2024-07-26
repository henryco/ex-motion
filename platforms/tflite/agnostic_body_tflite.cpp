//
// Created by henryco on 19/07/24.
//

#include <cstring>

#include "../agnostic_body.h"
#include <filesystem>
#include <fstream>

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

    AgnosticBody::AgnosticBody(const body::Model _model) {
        dnn_runner = platform::dnn::create();
        model = _model;

        const auto path = "./../models/blazepose/body/whole/" + body::models[model];
        if (!std::filesystem::exists(path))
            throw std::invalid_argument("Model file [" + path + "] cannot be located!");

        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            throw std::invalid_argument("Input file [" + path + "] cannot be located!");

        input.seekg(0, std::ios::end);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);
        auto buffer = new char[size];

        if (!input.read(buffer, size)) {
            delete[] buffer;
            throw std::runtime_error("Cannot read the file: " + path);
        }

        dnn_runner->init(buffer, size);
        delete[] buffer;
    }

    AgnosticBody::~AgnosticBody() {
        if (dnn_runner)
            delete dnn_runner;
        if (segmentations) {
            for (int i = 0; i < batch; i++)
                delete[] segmentations[i];
            delete[] segmentations;
        }
        if (landmarks_3d) {
            for (int i = 0; i < batch; i++)
                delete[] landmarks_3d[i];
            delete[] landmarks_3d;
        }
        if (pose_flags) {
            for (int i = 0; i < batch; i++)
                delete[] pose_flags[i];
            delete[] pose_flags;
        }
    }

    void AgnosticBody::inference(const size_t batch_size, const float *in_batch_ptr) {
        if (batch_size != batch) {

            if (segmentations != nullptr) {
                for (int i = 0; i < batch; i++)
                    delete[] segmentations[i];
                delete[] segmentations;
                segmentations = nullptr;
            }

            if (landmarks_3d != nullptr) {
                for (int i = 0; i < batch; i++)
                    delete[] landmarks_3d[i];
                delete[] landmarks_3d;
                landmarks_3d = nullptr;
            }

            if (pose_flags != nullptr) {
                for (int i = 0; i < batch; i++)
                    delete[] pose_flags[i];
                delete[] pose_flags;
                pose_flags = nullptr;
            }

            batch = batch_size;
        }

        const size_t size   = get_in_w() * get_in_h() * 3 * sizeof(float);
        const size_t n_seg  = get_n_segmentation_w() * get_n_segmentation_h();
        const size_t n_l3d = get_n_lm3d();

        if (segmentations == nullptr) {
            segmentations = new float *[batch];
            for (size_t i = 0; i < batch; i++)
                segmentations[i] = new float[n_seg * 1];
        }

        if (landmarks_3d == nullptr) {
            landmarks_3d = new float *[batch];
            for (size_t i = 0; i < batch; i++)
                landmarks_3d[i] = new float[n_l3d * 1];
        }

        if (pose_flags == nullptr) {
            pose_flags = new float *[batch];
            for (size_t i = 0; i < batch; i++)
                pose_flags[i] = new float[1];
        }

        for (size_t i = 0; i < batch; i++) {
            const float *ptr = &in_batch_ptr[i * get_in_w() * get_in_h() * 3];

            dnn_runner->buffer_f_input(0, size, ptr);
            dnn_runner->invoke();

            const auto ptr_seg = (float *) dnn_runner->buffer_f_output(body::mappings[model].seg);
            const auto ptr_l3d = (float *) dnn_runner->buffer_f_output(body::mappings[model].lm_3d);
            const auto ptr_flg = (float *) dnn_runner->buffer_f_output(body::mappings[model].flag);

            if (ptr_seg == nullptr || ptr_l3d == nullptr || ptr_flg == nullptr)
                throw std::runtime_error("Output result is nullptr");

            memcpy(&segmentations[i], ptr_seg, n_seg * 1 * sizeof(float));
            memcpy(&landmarks_3d[i], ptr_l3d, n_l3d * 1 * sizeof(float));
            memcpy(&pose_flags[i], ptr_flg, 1 * sizeof(float));
        }
    }

    const float * const * AgnosticBody::get_segmentations() const {
        return segmentations;
    }

    const float * const * AgnosticBody::get_landmarks_3d() const {
        return landmarks_3d;
    }

    const float * const * AgnosticBody::get_pose_flags() const {
        return pose_flags;
    }

    body::Model AgnosticBody::get_model() const {
        return model;
    }

    size_t AgnosticBody::get_in_w() const {
        return body::mappings[model].i_w;
    }

    size_t AgnosticBody::get_in_h() const {
        return body::mappings[model].i_h;
    }

    size_t AgnosticBody::get_n_lm3d() const {
        return 195;
    }

    size_t AgnosticBody::get_n_segmentation_w() const {
        return body::mappings[model].o_w;
    }

    size_t AgnosticBody::get_n_segmentation_h() const {
        return body::mappings[model].o_h;
    }

}
