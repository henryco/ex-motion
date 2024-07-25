//
// Created by henryco on 17/07/24.
//

#ifndef XMOTION_AGNOSTIC_DNN_H
#define XMOTION_AGNOSTIC_DNN_H

#include <vector>

namespace platform::dnn {

    class AgnosticDnn {
    public:
        virtual ~AgnosticDnn() = 0;

        virtual void init(const char *model, size_t size) = 0;

        virtual void reset() = 0;

        virtual void resize_input(int index, const std::vector<int> &size) = 0;

        /**
         * @param index index of the input tensor
         * @param in_batch_ptr FLATTENED 2D array into 1D
         */
        virtual void buffer_f_input(int index, size_t input_size, const void *in_batch_ptr) = 0;

        virtual void buffer_f_output(int index, size_t output_size, void *out_batch_ptr) = 0;

        virtual void* buffer_f_output(int index) = 0;

        virtual void invoke() = 0;
    };

    AgnosticDnn* create();
}

#endif //XMOTION_AGNOSTIC_DNN_H
