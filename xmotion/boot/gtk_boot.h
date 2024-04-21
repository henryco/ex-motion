//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_GTK_BOOT_H
#define XMOTION_GTK_BOOT_H

#include "i_boot.h"

namespace xm {

    class GtkBoot : public xm::Boot {
    public:
        int boot(int &argc, char **&argv) override;

        void open_project(const char *argv) override;
    };

} // xm

#endif //XMOTION_GTK_BOOT_H
