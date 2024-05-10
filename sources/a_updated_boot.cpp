//
// Created by henryco on 4/20/24.
//

#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "../xmotion/boot/a_updated_boot.h"

namespace xm {

    void UpdatedBoot::print_ocv_ocl_stats() {
        std::cout << "OpenCV version: " << CV_VERSION << '\n';

        if (!cv::ocl::useOpenCL())
            std::cerr << "OpenCL is not available..." << '\n';

        else {
            cv::ocl::Context context;
            if (!context.create(cv::ocl::Device::TYPE_GPU)) {
                std::cout << "Failed to create an OpenCL GPU context." << std::endl;
            } else {
                // Get the device(s) of the created context
                const auto total = context.ndevices();

                // Output device information
                for (size_t i = 0; i < total; i++) {
                    const auto &device = context.device(i);
                    std::cout << "GPU Device " << i << ": " << std::endl;
                    std::cout << " Name: " << device.name() << std::endl;
                    std::cout << " OpenCL C Version: " << device.OpenCL_C_Version() << std::endl;
                    std::cout << " OpenCL Version: " << device.OpenCLVersion() << std::endl;
                }
            }
        }
    }

    int UpdatedBoot::boot(int &argc, char **&argv) {
        cv::ocl::setUseOpenCL(true);

        print_ocv_ocl_stats();

//        deltaLoop.setFps(300);
        deltaLoop.setFps(0);
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