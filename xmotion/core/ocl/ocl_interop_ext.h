//
// Created by henryco on 13/06/24.
//

#ifndef XMOTION_OCL_INTEROP_EXT_H
#define XMOTION_OCL_INTEROP_EXT_H

#include <CL/cl.h>

namespace xm::ocl::iop {

    template <typename T>
    class CLPromise {
    protected:
        cl_command_queue ocl_queue = nullptr;
        cl_event ocl_event = nullptr;
        bool completed = false;
        T data;

    public:
        CLPromise<T>(T obj, cl_command_queue queue, cl_event event = nullptr):
                data(obj), ocl_queue(queue), ocl_event(event), completed(false) {}

        CLPromise<T>(T obj, cl_event event = nullptr): // NOLINT(*-explicit-constructor)
                data(obj), ocl_queue(nullptr), ocl_event(event), completed(true) {}

        CLPromise<T>() = default;

        ~CLPromise() {
            if (ocl_event != nullptr)
                clReleaseEvent(ocl_event);
        }

        CLPromise<T>(CLPromise<T> &&other) noexcept {
            completed = other.completed;
            ocl_queue = other.ocl_queue;
            ocl_event = other.ocl_event;
            data = std::move(other.data);
            other.ocl_event = nullptr;
            other.ocl_queue = nullptr;
            other.completed = true;
        }

        CLPromise<T>(const CLPromise<T> &other) {
            completed = other.completed;
            ocl_queue = other.ocl_queue;
            ocl_event = other.ocl_event;
            data = other.data;
            clRetainEvent(ocl_event);
        }

        CLPromise<T> &operator=(CLPromise<T> &&other) noexcept {
            if (this == &other)
                *this;
            if (ocl_event != nullptr)
                clReleaseEvent(ocl_event);
            completed = other.completed;
            ocl_queue = other.ocl_queue;
            ocl_event = other.ocl_event;
            data = std::move(other.data);
            other.ocl_event = nullptr;
            other.ocl_queue = nullptr;
            other.completed = true;
            return *this;
        }

        CLPromise<T> &operator=(const CLPromise<T> &other) {
            if (this == &other)
                *this;
            if (ocl_event != nullptr)
                clReleaseEvent(ocl_event);
            completed = other.completed;
            ocl_queue = other.ocl_queue;
            ocl_event = other.ocl_event;
            data = other.data;
            clRetainEvent(ocl_event);
            return *this;
        }

        /**
         * Often you should call waitFor() first
         */
        T get() {
            return data;
        }

        /**
         * Waits for data to be ready
         */
        CLPromise<T> &waitFor() {
            if (completed)
                return *this;
            {
                cl_int err;
                err = clFinish(ocl_queue);
                if (err != CL_SUCCESS)
                    throw std::runtime_error("Cannot finish command queue: " + std::to_string(err));
                completed = true;
            }
            return *this;
        }

        bool resolved() const {
            return completed;
        }

        const cl_event &event() const {
            return ocl_event;
        }

        cl_command_queue queue() const {
            return ocl_queue;
        }
    };

}

#endif //XMOTION_OCL_INTEROP_EXT_H
