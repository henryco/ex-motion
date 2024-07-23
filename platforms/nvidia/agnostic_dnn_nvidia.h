//
// Created by henryco on 18/07/24.
//

#ifndef XMOTION_AGNOSTIC_DNN_NVIDIA_H
#define XMOTION_AGNOSTIC_DNN_NVIDIA_H

#include "../agnostic_dnn.h"

namespace platform::dnn {

    class DnnInferenceNvidia : public AgnosticDnn {

    public:
        ~DnnInferenceNvidia() override;

        void init(const char *model, size_t size) override;

        void resize_input(int index, const std::vector<int> &size) override;

        void reset() override;

        void buffer_f_input(int index, size_t input_size, const float *batch_ptr) override;

        void buffer_f_output(int index, size_t batch_size, size_t output_size, float **out_batch_ptr) override;

        void buffer_f_output(int index, size_t output_size, float *out_batch_ptr) override;

        void invoke() override;
    };

} // platform

#endif //XMOTION_AGNOSTIC_DNN_NVIDIA_H
