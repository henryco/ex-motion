//
// Created by henryco on 4/16/24.
//

#include <iostream>
#include "../xmotion/boot/gtk_boot.h"

namespace xm {
    int GtkBoot::boot(int &argc, char **&argv) {
        // TODO GUI MODE
        std::cerr << "Right now, there is no support for full graphic mode" << '\n';
        return -1;
    }

    void GtkBoot::open_project(const char *argv) {

    }
} // xm