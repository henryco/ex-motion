//
// Created by henryco on 18/07/24.
//

#include "../agnostic_detector.h"
#include <filesystem>
#include <fstream>
#include <vector>

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
        if (bboxes)
            delete[] bboxes;
        if (scores)
            delete[] scores;
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

    void AgnosticDetector::inference(size_t batch_size, size_t input_size, const float * const*in_batch_ptr) {
        if (batch_size != batch) {

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

            const auto n_bboxes = get_n_bboxes();
            const auto n_scores = get_n_scores();

            bboxes = new float *[batch_size];
            scores = new float *[batch_size];

            for (int i = 0; i < batch_size; i++) {
                bboxes[i] = new float[n_bboxes];
                scores[i] = new float[n_scores];
            }

            batch = batch_size;
        }

        dnn_runner->buffer_f_input(0, batch_size, input_size, in_batch_ptr);
        dnn_runner->invoke();
    }


    void AgnosticDetector::inference(size_t batch_size, size_t input_size, const float *in_batch_ptr) {
        if (batch_size != batch) {

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

            const auto n_bboxes = get_n_bboxes();
            const auto n_scores = get_n_scores();

            bboxes = new float *[batch_size];
            scores = new float *[batch_size];

            for (int i = 0; i < batch_size; i++) {
                bboxes[i] = new float[n_bboxes];
                scores[i] = new float[n_scores];
            }

            batch = batch_size;
        }

        dnn_runner->buffer_f_input(0, batch_size, input_size, in_batch_ptr);
        dnn_runner->invoke();
    }

    const float *const *AgnosticDetector::get_bboxes() const {
        dnn_runner->buffer_f_output(detector::mappings[model].box_loc, batch, get_n_bboxes(), bboxes);
        return bboxes;
    }

    const float *const *AgnosticDetector::get_scores() const {
        dnn_runner->buffer_f_output(detector::mappings[model].score_loc, batch, get_n_scores(), scores);
        return scores;
    }

}