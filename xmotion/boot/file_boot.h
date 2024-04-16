//
// Created by henryco on 4/16/24.
//

#ifndef XMOTION_FILE_BOOT_H
#define XMOTION_FILE_BOOT_H

#include "boot.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace xm {

    class FileBoot : public xm::Boot {

        static inline const auto log =
                spdlog::stdout_color_mt("file_boot");

    protected:
        std::string project_dir;

    public:
        void boot(int &argc, char **&argv) override;

        void project(const char *argv) override;
    };

} // xm

#endif //XMOTION_FILE_BOOT_H
