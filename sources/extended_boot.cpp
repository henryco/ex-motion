//
// Created by henryco on 4/20/24.
//

#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "../xmotion/boot/extended_boot.h"

namespace xm {

    int ExtendedBoot::boot(int &argc, char **&argv) {
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

        const auto result = boostrap(argc, argv);
        if (result != 0)
            return result;

        deltaLoop.start();

        return result;
    }

    void ExtendedBoot::setTargetFps(int fps) {
        deltaLoop.setFps(fps);
    }
} // xm