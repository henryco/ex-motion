//
// Created by henryco on 11/06/24.
//

#include "../../xmotion/core/ocl/ocl_data.h"

namespace xm::ocl {

    void Image2D::copy_from(const Image2D &other) {
        detached = false;
        cols = other.cols;
        rows = other.rows;
        channels = other.channels;
        channel_size = other.channel_size;
        context = other.context;
        device = other.device;
        handle = other.handle;
    }

    void Image2D::reset_state(Image2D &other) {
        other.detached = false;
        other.context = nullptr;
        other.device = nullptr;
        other.handle = nullptr;
        other.channel_size = 0;
        other.channels = 0;
        other.cols = 0;
        other.rows = 0;
    }

    void Image2D::release() {
        if (handle != nullptr && !detached)
            clReleaseMemObject(handle);
    }

    std::size_t Image2D::size() const {
        return cols * rows * channels * channel_size;
    }

    bool Image2D::empty() const {
        return handle != nullptr;
    }

    Image2D::~Image2D() {
        release();
    }

    Image2D::Image2D(const Image2D &other) {
        copy_from(other);
    }

    Image2D::Image2D(Image2D &&other) noexcept {
        copy_from(other);
        reset_state(other);
    }

    Image2D &Image2D::detach(bool _) {
        detached = _;
        return *this;
    }

    Image2D &Image2D::operator=(const Image2D &other) {
        if (this == &other)
            return *this;
        release();
        copy_from(other);
        return *this;
    }

    Image2D &Image2D::operator=(Image2D &&other) noexcept {
        if (this == &other)
            return *this;
        release();
        copy_from(other);
        reset_state(other);
        return *this;
    }

    Image2D::Image2D(size_t _cols,
                     size_t _rows,
                     size_t _channels,
                     size_t _channel_size,
                     cl_mem _handle,
                     cl_context _context,
                     cl_device_id _device)
            : cols(_cols),
              rows(_rows),
              channels(_channels),
              channel_size(_channel_size),
              handle(_handle),
              context(_context),
              device(_device),
              detached(false) {}
}