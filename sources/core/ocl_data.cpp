//
// Created by henryco on 11/06/24.
//

#include <stdexcept>
#include "../../xmotion/core/ocl/ocl_data.h"

namespace xm::ocl {

    Image2D &Image2D::decrement_ref() {
        if (handle != nullptr)
            clReleaseMemObject(handle);
        return *this;
    }

    void Image2D::release() {
        if (handle != nullptr && !is_detached)
            clReleaseMemObject(handle); // TODO FIXME 4
        reset_state(*this);
    }

    std::size_t Image2D::size() const {
        return cols * rows * channels * channel_size;
    }

    bool Image2D::empty() const {
        return handle == nullptr || size() == 0;
    }

    cl_mem Image2D::get_handle(ACCESS desired) const {
        if ((static_cast<int>(desired) & static_cast<int>(access)) != static_cast<int>(desired))
            throw std::invalid_argument("CL mem access denied");
        return handle;
    }

    cl_mem &Image2D::get_handle(ACCESS desired) {
        if ((static_cast<int>(desired) & static_cast<int>(access)) != static_cast<int>(desired))
            throw std::invalid_argument("CL mem access denied");
        return handle;
    }

    Image2D::~Image2D() {
        release();
    }

    void Image2D::copy_from(const Image2D &other) {
        is_detached = false;
        access = other.access;
        cols = other.cols;
        rows = other.rows;
        channels = other.channels;
        channel_size = other.channel_size;
        context = other.context;
        device = other.device;
        handle = other.handle;
    }

    void Image2D::reset_state(Image2D &other) {
        other.is_detached = true;
        other.context = nullptr;
        other.device = nullptr;
        other.handle = nullptr;
        other.channel_size = 0;
        other.channels = 0;
        other.cols = 0;
        other.rows = 0;
    }

    Image2D::Image2D(const Image2D &other) {
        copy_from(other);
        clRetainMemObject(handle);
    }

    Image2D::Image2D(const Image2D &other, cl_mem _handle) {
        copy_from(other);
        handle = _handle;
    }

    Image2D::Image2D(const Image2D &other, cl_mem _handle, ACCESS modifier) {
        copy_from(other);
        handle = _handle;
        access = modifier;
    }

    Image2D::Image2D(Image2D &&other) noexcept {
        copy_from(other);
        reset_state(other);
    }

    Image2D::Image2D(const Image2D &&other, cl_mem _handle) {
        copy_from(other);
        handle = _handle;
    }

    Image2D::Image2D(const Image2D &&other, cl_mem _handle, ACCESS modifier) {
        copy_from(other);
        handle = _handle;
        access = modifier;
    }

    Image2D &Image2D::detached(bool _) {
        is_detached = _;
        return *this;
    }

    Image2D &Image2D::retain() {
        clRetainMemObject(handle);
        return *this;
    }

    Image2D &Image2D::operator=(const Image2D &other) {
        if (this == &other)
            return *this;
        release();
        copy_from(other);
        clRetainMemObject(handle);
        return *this;
    }

    Image2D &Image2D::operator=(Image2D &&other) noexcept {
        if (this == &other)
            return *this;
        release(); // TODO FIXME 3
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
                     cl_device_id _device,
                     ACCESS _access)
            : cols(_cols),
              rows(_rows),
              channels(_channels),
              channel_size(_channel_size),
              handle(_handle),
              context(_context),
              device(_device),
              access(_access),
              is_detached(false) {}

    Image2D::Image2D() {
        reset_state(*this);
    }

    Image2D Image2D::allocate(size_t cols, size_t rows, size_t channels, size_t channel_size, cl_context context,
                              cl_device_id device, ACCESS access) {
        cl_mem_flags flags;
        if (access == ACCESS::RO) flags = CL_MEM_READ_ONLY;
        else if (access == ACCESS::WO) flags = CL_MEM_WRITE_ONLY;
        else if (access == ACCESS::RW) flags = CL_MEM_READ_WRITE;
        else throw std::invalid_argument("Invalid access modifier: " + std::to_string((int) access));

        cl_int err;
        cl_mem buffer = clCreateBuffer(context, flags, cols * rows * channels * channel_size, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("Cannot create cl buffer: " + std::to_string(err));
        return Image2D(cols, rows, channels, channel_size, buffer, context, device, access);
    }

    Image2D Image2D::allocate_like(const Image2D &t, ACCESS access) {
        return allocate(t.cols, t.rows, t.channels, t.channel_size, t.context, t.device, access);
    }
}