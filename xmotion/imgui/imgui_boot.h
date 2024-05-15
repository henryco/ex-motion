//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_IMGUI_BOOT_H
#define XMOTION_IMGUI_BOOT_H

#include "../core/boot/i_boot.h"

namespace xm {

    class IMGuiBoot : public xm::Boot {
    public:
        int boot(int &argc, char **&argv) override;

        void open_project(const char *argv) override;
    };

} // xm

#endif //XMOTION_IMGUI_BOOT_H
