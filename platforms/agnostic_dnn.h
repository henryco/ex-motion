//
// Created by henryco on 17/07/24.
//

#ifndef XMOTION_AGNOSTIC_DNN_H
#define XMOTION_AGNOSTIC_DNN_H

#include <string>

namespace platform::dnn {

    class AgnosticDnn {
    public:
        virtual ~AgnosticDnn() = 0;

        virtual void init(const char *model, size_t size) = 0;

        virtual void reset() = 0;

        /**
         * @param index index of the input tensor
         * @param in_batch_ptr 2D array
         */
        virtual void buffer_f_input(int index, size_t batch_size, size_t input_size, const float * const*in_batch_ptr) = 0;

        /**
         * @param index index of the input tensor
         * @param in_batch_ptr FLATTENED 2D array into 1D
         */
        virtual void buffer_f_input(int index, size_t batch_size, size_t input_size, const float *in_batch_ptr) = 0;

        virtual void buffer_f_output(int index, size_t batch_size, size_t output_size, float **out_batch_ptr) = 0;

        virtual void invoke() = 0;
    };

    AgnosticDnn* create();
}

#endif //XMOTION_AGNOSTIC_DNN_H
