//
// Created by henryco on 4/20/24.
//

#ifndef XMOTION_EXTENDED_BOOT_H
#define XMOTION_EXTENDED_BOOT_H

#include "boot.h"
#include "../utils/delta_loop.h"

namespace xm {

    class ExtendedBoot : public xm::Boot {

    protected:
        eox::util::DeltaLoop deltaLoop;

    public:

        int boot(int &argc, char **&argv) final;

        virtual void setTargetFps(int fps) final;

        virtual int boostrap(int &argc, char **&argv) = 0;

        virtual void update(float delta, float latency, float fps) = 0;
    };

} // xm

#endif //XMOTION_EXTENDED_BOOT_H
