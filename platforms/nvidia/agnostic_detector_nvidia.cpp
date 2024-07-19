//
// Created by henryco on 19/07/24.
//

#include "../agnostic_detector.h"

namespace platform::dnn::detector {
    const Metadata mappings[3] = {
        {},
        {},
        {}
    };

    const std::string models[3] = {
        "",
        "",
        ""
    };
}

namespace platform::dnn {

    AgnosticDetector::AgnosticDetector(detector::Model _model) {
        model = _model;
    }

    AgnosticDetector::~AgnosticDetector() {

    }

    detector::Model AgnosticDetector::get_model() const {
        return model;
    }

    size_t AgnosticDetector::get_in_w() const {
        return 0;
    }

    size_t AgnosticDetector::get_in_h() const {
        return 0;
    }

    int AgnosticDetector::get_n_bboxes() const {
        return 0;
    }

    int AgnosticDetector::get_n_scores() const {
        return 0;
    }

    void AgnosticDetector::inference(size_t batch_size, size_t input_size, const float * const *in_batch_ptr) {
    }

    void AgnosticDetector::inference(size_t batch_size, size_t input_size, const float *in_batch_ptr) {
    }

    const float * const * AgnosticDetector::get_bboxes() const {
        return nullptr;
    }

    const float * const * AgnosticDetector::get_scores() const {
        return nullptr;
    }

}
