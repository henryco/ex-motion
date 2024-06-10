//
// Created by henryco on 11/06/24.
//

#ifndef XMOTION_OCL_DATA_H
#define XMOTION_OCL_DATA_H

#include <CL/cl.h>

namespace xm::ocl {

    class Image2D {

    public:
        std::size_t cols = 0;
        std::size_t rows = 0;
        std::size_t channels = 0;
        std::size_t channel_size = 0;

        cl_mem handle = nullptr;
        cl_context context = nullptr;
        cl_device_id device = nullptr;

        bool detached = false;

        std::size_t size() const;

        bool empty() const;

        Image2D(size_t cols,
                size_t rows,
                size_t channels,
                size_t channel_size,
                cl_mem handle,
                cl_context context,
                cl_device_id device);

        Image2D(const Image2D &other);

        Image2D(Image2D &&other) noexcept;

        Image2D& operator=(const Image2D& other);

        Image2D& operator=(Image2D&& other) noexcept;

        ~Image2D();

        /**
         * !!! May cause memory leaks !!!
         */
        Image2D &detach(bool _ = true);

    private:
        void copy_from(const Image2D &other);

        void reset_state(Image2D &other);

        void release();
    };

}

#endif //XMOTION_OCL_DATA_H
