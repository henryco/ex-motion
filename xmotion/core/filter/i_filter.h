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
        virtual xm::ocl::iop::ClImagePromise filter(const xm::ocl::Image2D &in, int q_idx) = 0;

        xm::ocl::iop::ClImagePromise filter(const xm::ocl::Image2D &in) {return filter(in, -1);}

        virtual ~Filter() = default;
    };

} // xm

#endif //XMOTION_I_FILTER_H
