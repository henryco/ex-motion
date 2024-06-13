//
// Created by henryco on 4/21/24.
//

#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-use-nodiscard"

#ifndef XMOTION_LOGIC_H
#define XMOTION_LOGIC_H

#include <vector>
#include "../ocl/ocl_data.h"

namespace xm {

    class Logic {

    public:
        virtual ~Logic() = default;

        virtual Logic& proceed(float delta, const std::vector<xm::ocl::Image2D> &frames) = 0;

        virtual const std::vector<xm::ocl::Image2D> &frames() const = 0;

        virtual bool is_active() const = 0;

        virtual void start() = 0;

        virtual void stop() = 0;

        virtual void debug(bool _debug) = 0;
    };

} // xm

#endif //XMOTION_LOGIC_H

#pragma clang diagnostic pop