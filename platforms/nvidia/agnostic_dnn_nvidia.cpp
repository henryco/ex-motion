//
// Created by henryco on 18/07/24.
//

#include "agnostic_dnn_nvidia.h"

namespace platform::dnn {

    AgnosticDnn::~AgnosticDnn() {}

    AgnosticDnn *create() {
        return new DnnInferenceNvidia();
    }

    DnnInferenceNvidia::~DnnInferenceNvidia() {

    }

    void DnnInferenceNvidia::init(const char *model, size_t size) {

    }

    void DnnInferenceNvidia::resize_input(int index, const std::vector<int> &size) {
    }

    void DnnInferenceNvidia::reset() {

    }

    void DnnInferenceNvidia::buffer_f_input(int index, size_t input_size, const float *batch_ptr) {

    }

    void DnnInferenceNvidia::buffer_f_output(int index, size_t batch_size, size_t output_size, float **out_batch_ptr) {

    }

    void DnnInferenceNvidia::buffer_f_output(int index, size_t output_size, float *out_batch_ptr) {
    }

    void DnnInferenceNvidia::invoke() {

    }
} // platform