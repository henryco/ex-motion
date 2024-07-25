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

    void AgnosticBody::inference(const size_t batch_size, const float *in_batch_ptr) {
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

            segmentations = new float*[batch_size];
            heatmaps      = new float*[batch_size];
            landmarks_3d  = new float*[batch_size];
            landmarks_wd  = new float*[batch_size];
            pose_flags    = new float*[batch_size];

            dnn_runner->resize_input(0, {(int) batch_size, (int) get_in_w(), (int) get_in_h(), 3});
            batch = batch_size;
        }

        const auto size = get_in_w() * get_in_h() * 3 * sizeof(float);
        dnn_runner->buffer_f_input(0, size, in_batch_ptr);
        dnn_runner->invoke();

        const auto ptr_seg = (float *) dnn_runner->buffer_f_output(body::mappings[model].seg);
        const auto ptr_hmp = (float *) dnn_runner->buffer_f_output(body::mappings[model].hm);
        const auto ptr_l3d = (float *) dnn_runner->buffer_f_output(body::mappings[model].lm_3d);
        const auto ptr_lwd = (float *) dnn_runner->buffer_f_output(body::mappings[model].world);
        const auto ptr_flg = (float *) dnn_runner->buffer_f_output(body::mappings[model].flag);

        if (ptr_seg == nullptr || ptr_hmp == nullptr || ptr_l3d == nullptr || ptr_lwd == nullptr || ptr_flg == nullptr)
            throw std::runtime_error("Output result is nullptr");

        if (!segmentations)
            segmentations = new float *[batch];

        if (!landmarks_3d)
            landmarks_3d = new float *[batch];

        if (!landmarks_wd)
            landmarks_wd = new float *[batch];

        if (!pose_flags)
            pose_flags = new float *[batch];

        if (!heatmaps)
            heatmaps = new float *[batch];

        const auto n_seg     = get_n_segmentation_w() * get_n_segmentation_h();
        const auto n_heatmap = get_n_heatmap_w() * get_n_heatmap_h();
        const auto n_lm3d    = get_n_lm3d();
        const auto n_lmwd    = get_n_lmwd();

        for (int i = 0; i < batch; i++) {
            heatmaps[i]      = &ptr_hmp[i * batch * n_heatmap * 39];
            landmarks_3d[i]  = &ptr_l3d[i * batch * n_lm3d * 1];
            landmarks_wd[i]  = &ptr_lwd[i * batch * n_lmwd * 1];
            segmentations[i] = &ptr_seg[i * batch * n_seg * 1];
            pose_flags[i]    = &ptr_flg[i * batch * 1];
        }
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

    size_t AgnosticBody::get_n_lmwd() const {
        return 117;
    }

    size_t AgnosticBody::get_n_segmentation_w() const {
        return body::mappings[model].o_w;
    }

    size_t AgnosticBody::get_n_segmentation_h() const {
        return body::mappings[model].o_h;
    }

    size_t AgnosticBody::get_n_heatmap_w() const {
        return 64;
    }

    size_t AgnosticBody::get_n_heatmap_h() const {
        return 64;
    }

}
