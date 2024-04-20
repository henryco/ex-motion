//
// Created by henryco on 4/20/24.
//

#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "../xmotion/boot/updated_boot.h"

namespace xm {

    int UpdatedBoot::boot(int &argc, char **&argv) {
        std::cout << "OpenCV version: " << CV_VERSION << '\n';
        cv::ocl::setUseOpenCL(true);
        if (!cv::ocl::useOpenCL()) {
            std::cerr << "OpenCL is not available..." << '\n';
            return 1;
        }

        deltaLoop.setFps(300);
        deltaLoop.setFunc([this](float d, float l, float f) {
            update(d, l, f);
        });
        return boostrap(argc, argv);
    }

    void UpdatedBoot::set_loop_fps(int fps) {
        deltaLoop.setFps(fps);
    }

    void UpdatedBoot::start_loop() {
        deltaLoop.start();
    }

    void UpdatedBoot::start_loop(int fps) {
        set_loop_fps(fps);
        start_loop();
    }

    void UpdatedBoot::stop_loop() {
        deltaLoop.stop();
    }
} // xm