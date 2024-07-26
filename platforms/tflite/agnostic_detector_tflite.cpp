//
// Created by henryco on 18/07/24.
//

#include "../agnostic_detector.h"
#include <filesystem>
#include <fstream>
#include <cstring>

namespace platform::dnn {

    namespace detector {

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

    AgnosticDetector::AgnosticDetector(detector::Model _model) {
        dnn_runner = platform::dnn::create();
        model = _model;

        const auto path = "./../models/blazepose/body/detector/" + detector::models[model];
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

    AgnosticDetector::~AgnosticDetector() {
        if (dnn_runner)
            delete dnn_runner;
        if (bboxes) {
            for (int i = 0; i < batch; i++)
                delete[] bboxes[i];
            delete[] bboxes;
        }
        if (scores) {
            for (int i = 0; i < batch; i++)
                delete[] scores[i];
            delete[] scores;
        }
    }

    detector::Model AgnosticDetector::get_model() const {
        return model;
    }

    size_t AgnosticDetector::get_in_w() const {
        return detector::mappings[model].i_w;
    }

    size_t AgnosticDetector::get_in_h() const {
        return detector::mappings[model].i_h;
    }

    int AgnosticDetector::get_n_bboxes() const {
        return detector::mappings[model].bboxes;
    }

    int AgnosticDetector::get_n_scores() const {
        return detector::mappings[model].scores;
    }

    void AgnosticDetector::inference(size_t batch_size, const float *in_batch_ptr) {
        if (batch_size != batch) {

            if (bboxes != nullptr) {
                for (int i = 0; i < batch; i++)
                    delete[] bboxes[i];
                delete[] bboxes;
                bboxes = nullptr;
            }

            if (scores != nullptr) {
                for (int i = 0; i < batch; i++)
                    delete[] scores[i];
                delete[] scores;
                scores = nullptr;
            }

            batch = batch_size;
        }

        const size_t dim      = get_in_w() * get_in_h() * 3;
        const size_t n_bboxes = get_n_bboxes();
        const size_t n_scores = get_n_scores();

        if (bboxes == nullptr) {
            bboxes = new float *[batch];
            for (size_t i = 0; i < batch; i++)
                bboxes[i] = new float[n_bboxes * 12];
        }

        if (scores == nullptr) {
            scores = new float *[batch];
            for (size_t i = 0; i < batch; i++)
                scores[i] = new float[n_scores * 1];
        }

        for (size_t i = 0; i < batch; i++) {
            const float *ptr = &in_batch_ptr[i * dim];

            dnn_runner->buffer_f_input(0, dim * sizeof(float), ptr);
            dnn_runner->invoke();

            const auto ptr_box = (float*) dnn_runner->buffer_f_output(detector::mappings[model].box_loc);
            const auto ptr_scr = (float*) dnn_runner->buffer_f_output(detector::mappings[model].score_loc);

            if (ptr_box == nullptr || ptr_scr == nullptr)
                throw std::runtime_error("Output result is nullptr");

            for (size_t k = 0; k < n_bboxes * 12; k++)
                bboxes[i][k] = ptr_box[k];
            for (size_t k = 0; k < n_scores * 1; k++)
                scores[i][k] = ptr_scr[k];

            // WTF MEMCPY CAUSING GPU DELEGATE (CL) CRASH ????? !!!!
            // std::memcpy(&bboxes[i], ptr_box, n_bboxes * 12 * sizeof(float));
            // std::memcpy(&scores[i], ptr_scr, n_scores * 1 * sizeof(float));
        }
    }

    const float *const *AgnosticDetector::get_bboxes() const {
        return bboxes;
    }

    const float *const *AgnosticDetector::get_scores() const {
        return scores;
    }

}