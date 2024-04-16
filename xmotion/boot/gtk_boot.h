//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_GTK_BOOT_H
#define XMOTION_GTK_BOOT_H

#include "boot.h"

namespace xm {

    class GtkBoot : public xm::Boot {
    public:
        void boot(int &argc, char **&argv) override;

        void project(const char *argv) override;
    };

} // xm

#endif //XMOTION_GTK_BOOT_H
