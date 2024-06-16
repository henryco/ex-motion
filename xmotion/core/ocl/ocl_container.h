//
// Created by henryco on 16/06/24.
//

#ifndef XMOTION_OCL_CONTAINER_H
#define XMOTION_OCL_CONTAINER_H

#include <functional>

namespace xm::ocl {
    class ResourceContainer {
    private:
        std::function<void()> *cleanup_cb;
        bool released = false;
    public:
        explicit ResourceContainer(std::function<void()> *cb_ptr);

        ResourceContainer(const ResourceContainer &other) = delete;

        ResourceContainer(ResourceContainer &&other) = delete;

        ResourceContainer &operator=(ResourceContainer &&other) noexcept = delete;

        ResourceContainer &operator=(const ResourceContainer &other) = delete;

        ~ResourceContainer();

        void release();
    };
}

#endif //XMOTION_OCL_CONTAINER_H
