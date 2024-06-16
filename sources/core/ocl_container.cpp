//
// Created by henryco on 16/06/24.
//

#include "../../xmotion/core/ocl/ocl_container.h"

namespace xm::ocl {
    ResourceContainer::~ResourceContainer() {
        release();
    }

    ResourceContainer::ResourceContainer(std::function<void()> *cb_ptr) {
        cleanup_cb = cb_ptr;
        released = false;
    }

    void ResourceContainer::release() {
        if (released)
            return;
        released = true;
        if (cleanup_cb) {
            (*cleanup_cb)();
            delete cleanup_cb;
        }
    }
}
