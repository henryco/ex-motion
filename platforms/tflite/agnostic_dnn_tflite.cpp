//
// Created by henryco on 17/07/24.
//

#include "agnostic_dnn_tflite.h"

#include <iostream>
#include <tensorflow/lite/delegates/gpu/delegate.h>

namespace platform::dnn {

    AgnosticDnn::~AgnosticDnn() {}

    AgnosticDnn *create() {
        return new DnnInferenceTfLite();
    }

    DnnInferenceTfLite::~DnnInferenceTfLite() {
        if (gpu_delegate)
            TfLiteGpuDelegateV2Delete(gpu_delegate);
        if (model_buffer)
            delete[] model_buffer;
    }

    void DnnInferenceTfLite::reset() {
        if (gpu_delegate)
            TfLiteGpuDelegateV2Delete(gpu_delegate);
        if (model_buffer)
            delete[] model_buffer;
        initialized = false;
    }

    void DnnInferenceTfLite::init(const char *_model_buffer, size_t size) {
        if (initialized)
            return;

        model_buffer = new char[size];
        for (int i = 0; i < size; i++)
            model_buffer[i] = _model_buffer[i];
        model = std::move(tflite::FlatBufferModel::BuildFromBuffer(model_buffer, size));

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            throw std::runtime_error("Failed to create tflite interpreter");
        }

        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        gpu_delegate = TfLiteGpuDelegateV2Create(&options);

        if (interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk)
            throw std::runtime_error("Failed to modify graph with GPU delegate");
        if (interpreter->AllocateTensors() != kTfLiteOk)
            throw std::runtime_error("Failed to allocate tensors for tflite interpreter");

        initialized = true;
    }

    void DnnInferenceTfLite::resize_input(int index, const std::vector<int> &size) {
        const auto input_index = interpreter->inputs()[index];
        TfLiteStatus status = interpreter->ResizeInputTensor(input_index, size);
        if (status != kTfLiteOk)
            throw std::runtime_error("Failed to resize input tensors for tflite interpreter, status: " + std::to_string(status));

        status = interpreter->AllocateTensors();
        if (status != kTfLiteOk)
            throw std::runtime_error("Failed to allocate tensors for tflite interpreter, status: "+ std::to_string(status));
    }

    void DnnInferenceTfLite::buffer_f_input(int index, size_t input_size, const float *batch_ptr) {
        auto *input = interpreter->input_tensor(index)->data.f;
        std::memcpy(input, batch_ptr, input_size * sizeof(float));
    }

    void DnnInferenceTfLite::buffer_f_output(int index, size_t batch_size, size_t output_size, float **out_batch_ptr) {
        auto output_tensor = interpreter->output_tensor(index);
        const float *output = output_tensor->data.f;

        for (int i = 0; i < batch_size; i++)
            std::memcpy(out_batch_ptr[i], output + (i * output_size), output_size * sizeof(float));
    }

    void DnnInferenceTfLite::buffer_f_output(int index, size_t output_size, float *out_batch_ptr) {
        auto output_tensor = interpreter->output_tensor(index);
        const float *output = output_tensor->data.f;
        std::memcpy(out_batch_ptr, output, output_size * sizeof(float));
    }

    void DnnInferenceTfLite::invoke() {
        const auto status = interpreter->Invoke();
        if (status != kTfLiteOk)
            throw std::runtime_error("TFLite inference invocation error: " + std::to_string(status));
    }

}
