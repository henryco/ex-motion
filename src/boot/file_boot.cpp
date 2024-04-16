//
// Created by henryco on 4/16/24.
//

#include <filesystem>
#include "file_boot.h"

namespace xm {

    void FileBoot::project(const char *argv) {
        project_dir = std::string(argv);
        if (std::filesystem::exists(project_dir) && std::filesystem::is_directory(project_dir)) {
            log->info("Project: {}", project_dir);
        } else if (std::filesystem::exists(project_dir) && !std::filesystem::is_directory(project_dir)) {
            log->error("Project path exists but its not a directory");
            std::exit(1);
        } else if (std::filesystem::create_directories(project_dir)) {
            log->info("Project directory created: {}", project_dir);
        }
    }

    void FileBoot::boot(int &argc, char **&argv) {

    }



} // xm