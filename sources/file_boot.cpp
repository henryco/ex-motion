//
// Created by henryco on 4/16/24.
//

#include <filesystem>
#include "../xmotion/boot/file_boot.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <gtkmm/application.h>

namespace xm {

    void FileBoot::project(const char *argv) {
        project_conf = std::string(argv);

        if (std::filesystem::exists(project_conf) && std::filesystem::is_directory(project_conf)) {
            std::filesystem::path root = project_conf;
            std::filesystem::path file = "config.json";
            project_conf = (root / file).string();
        }

        if (!std::filesystem::exists(project_conf)) {
            log->error("Cannot locate: {}", project_conf);
            std::exit(1);
        }
    }

    void FileBoot::boot(int &argc, char **&argv) {

        {
            // init opencl
            cv::ocl::setUseOpenCL(true);
            if (!cv::ocl::useOpenCL()) {
                std::cerr << "OpenCL is not available..." << '\n';
                std::exit(1);
            }
            std::cout << "OpenCV version: " << CV_VERSION << '\n';
        }

        int n_argc = 1;
        const auto app = Gtk::Application::create(
                n_argc,
                argv,
                "dev.tindersamurai.xmotion"
        );

        // TODO
        app->run();
    }



} // xm