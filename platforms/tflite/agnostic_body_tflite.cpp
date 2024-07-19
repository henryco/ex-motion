//
// Created by henryco on 19/07/24.
//

#include "../agnostic_body.h"
#include <filesystem>
#include <fstream>

namespace platform::dnn {

    namespace body {
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
        if (segmentations)
            delete[] segmentations;
        if (landmarks_3d)
            delete[] landmarks_3d;
        if (landmarks_wd)
            delete[] landmarks_wd;
        if (pose_flags)
            delete[] pose_flags;
        if (heatmaps)
            delete[] heatmaps;
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

    size_t AgnosticBody::get_out_w() const {
        return body::mappings[model].o_w;
    }

    size_t AgnosticBody::get_out_h() const {
        return body::mappings[model].o_h;
    }

    size_t AgnosticBody::get_n_lm3d() const {
        return 195;
    }

    size_t AgnosticBody::get_n_lmwd() const {
        return 117;
    }

    size_t AgnosticBody::get_n_heatmap() const {
        return 64 * 64;
    }

    size_t AgnosticBody::get_n_segmentation() const {
        return 256 * 256;
    }

    void AgnosticBody::inference(const size_t batch_size, const size_t input_size, const float * const *in_batch_ptr) {
        if (batch_size != batch) {

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

            if (landmarks_wd) {
                for (int i = 0; i < batch; i++)
                    delete[] landmarks_wd[i];
                delete[] landmarks_wd;
            }

            if (pose_flags) {
                for (int i = 0; i < batch; i++)
                    delete[] pose_flags[i];
                delete[] pose_flags;
            }

            if (heatmaps) {
                for (int i = 0; i < batch; i++)
                    delete[] heatmaps[i];
                delete[] heatmaps;
            }

            const auto n_seg     = get_n_segmentation();
            const auto n_heatmap = get_n_heatmap();
            const auto n_lm3d    = get_n_lm3d();
            const auto n_lmwd    = get_n_lmwd();

            segmentations = new float*[batch_size];
            heatmaps      = new float*[batch_size];
            landmarks_3d  = new float*[batch_size];
            landmarks_wd  = new float*[batch_size];
            pose_flags    = new float*[batch_size];

            for (int i = 0; i < batch_size; i++) {
                segmentations[i] = new float[n_seg];
                heatmaps[i]      = new float[n_heatmap];
                landmarks_3d[i]  = new float[n_lm3d];
                landmarks_wd[i]  = new float[n_lmwd];
                pose_flags[i]    = new float[1];
            }

            batch = batch_size;
        }

        dnn_runner->buffer_f_input(0, batch_size, input_size, in_batch_ptr);
        dnn_runner->invoke();
    }

    void AgnosticBody::inference(const size_t batch_size, const size_t input_size, const float *in_batch_ptr) {
        if (batch_size != batch) {

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

            if (landmarks_wd) {
                for (int i = 0; i < batch; i++)
                    delete[] landmarks_wd[i];
                delete[] landmarks_wd;
            }

            if (pose_flags) {
                for (int i = 0; i < batch; i++)
                    delete[] pose_flags[i];
                delete[] pose_flags;
            }

            if (heatmaps) {
                for (int i = 0; i < batch; i++)
                    delete[] heatmaps[i];
                delete[] heatmaps;
            }

            const auto n_seg     = get_n_segmentation();
            const auto n_heatmap = get_n_heatmap();
            const auto n_lm3d    = get_n_lm3d();
            const auto n_lmwd    = get_n_lmwd();

            segmentations = new float*[batch_size];
            heatmaps      = new float*[batch_size];
            landmarks_3d  = new float*[batch_size];
            landmarks_wd  = new float*[batch_size];
            pose_flags    = new float*[batch_size];

            for (int i = 0; i < batch_size; i++) {
                segmentations[i] = new float[n_seg];
                heatmaps[i]      = new float[n_heatmap];
                landmarks_3d[i]  = new float[n_lm3d];
                landmarks_wd[i]  = new float[n_lmwd];
                pose_flags[i]    = new float[1];
            }

            batch = batch_size;
        }

        dnn_runner->buffer_f_input(0, batch_size, input_size, in_batch_ptr);
        dnn_runner->invoke();
    }

    const float * const * AgnosticBody::get_segmentations() const {
        return segmentations;
    }

    const float * const * AgnosticBody::get_landmarks_3d() const {
        return landmarks_3d;
    }

    const float * const * AgnosticBody::get_landmarks_wd() const {
        return landmarks_wd;
    }

    const float * const * AgnosticBody::get_pose_flags() const {
        return pose_flags;
    }

    const float * const * AgnosticBody::get_heatmaps() const {
        return heatmaps;
    }


}
