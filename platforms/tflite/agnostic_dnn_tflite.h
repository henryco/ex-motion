//
// Created by henryco on 18/07/24.
//

#ifndef XMOTION_AGNOSTIC_DNN_TFLITE_H
#define XMOTION_AGNOSTIC_DNN_TFLITE_H

#include "../agnostic_dnn.h"

#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/delegates/gpu/delegate_options.h>
#include <filesystem>

namespace platform::dnn {

    class DnnInferenceTfLite : public AgnosticDnn {

    private:
        std::unique_ptr<tflite::Interpreter> interpreter = nullptr;
        std::unique_ptr<tflite::FlatBufferModel> model = nullptr;
        TfLiteDelegate* gpu_delegate = nullptr;
        char *model_buffer = nullptr;

        bool initialized = false;

    public:
        ~DnnInferenceTfLite() override;

        void init(const char *model, size_t size) override;

        void reset() override;

        void resize_input(int index, const std::vector<int> &size) override;

        void buffer_f_input(int index, size_t size, const void *batch_ptr) override;

        void buffer_f_output(int index, size_t output_size, void *out_batch_ptr) override;

        void * buffer_f_output(int index) override;

        void invoke() override;
    };
}

#endif //XMOTION_AGNOSTIC_DNN_TFLITE_H
