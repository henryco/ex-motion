//
// Created by henryco on 6/2/24.
//

#ifndef XMOTION_I_FILTER_H
#define XMOTION_I_FILTER_H

#include "../ocl/ocl_data.h"
#include "../ocl/ocl_interop.h"

namespace xm {

    class Filter {
    public:
        virtual xm::ocl::iop::ClImagePromise filter(const xm::ocl::iop::ClImagePromise &in, int q_idx) = 0;

        xm::ocl::iop::ClImagePromise filter(const xm::ocl::iop::ClImagePromise &in) {return filter(in, -1);}

        virtual void reset() = 0;

        virtual void start() = 0;

        virtual void stop() = 0;

        virtual ~Filter() = default;
    };

} // xm

#endif //XMOTION_I_FILTER_H
