//
// Created by henryco on 4/20/24.
//

#ifndef XMOTION_UPDATED_BOOT_H
#define XMOTION_UPDATED_BOOT_H

#include "boot.h"
#include "../utils/delta_loop.h"

namespace xm {

    class UpdatedBoot : public xm::Boot {

    protected:
        eox::util::DeltaLoop deltaLoop;

    public:

        int boot(int &argc, char **&argv) final;

        virtual void set_loop_fps(int fps) final;

        virtual void start_loop(int fps) final;

        virtual void start_loop() final;

        virtual void stop_loop() final;

        virtual int boostrap(int &argc, char **&argv) = 0;

        virtual void update(float delta, float latency, float fps) = 0;

    private:
        static void print_ocv_ocl_stats();
    };

} // xm

#endif //XMOTION_UPDATED_BOOT_H
