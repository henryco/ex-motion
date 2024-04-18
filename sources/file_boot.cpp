//
// Created by henryco on 4/16/24.
//

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <gtkmm/application.h>
#include <fstream>

#include "../xmotion/boot/file_boot.h"
#include "../xmotion/gtk/small_button.h"

namespace xm {

    void FileBoot::update(float delta, float _, float fps) {
//        log->info("update: {}, {}", delta, fps);
        window->refresh();
    }

    void FileBoot::project(const char *argv) {
        project_path = xm::data::prepare_project_file(argv);
        config = xm::data::config_from_file(project_path);
    }

    void FileBoot::prepare_ocv() { // NOLINT(*-convert-member-functions-to-static)
        cv::ocl::setUseOpenCL(true);
        if (!cv::ocl::useOpenCL()) {
            log->error("OpenCL is not available...");
            std::exit(1);
        }
        log->info("OpenCV version: {}", CV_VERSION);
    }

    void FileBoot::prepare_gui() {
        auto button = Gtk::make_managed<xm::SmallButton>("c");
        button->proxy().signal_clicked().connect([this](){
            log->info("click");
        });

        window = std::make_unique<xm::SimpleImageWindow>();
        window->init(640, 480, {"test1", "test2"});
        window->add_one(*button);
        window->scale(1);
        window->show_all_children();
    }

    void FileBoot::prepare_loop() {
        deltaLoop.setFunc([this](float d, float l, float f) {
            update(d, l, f);
        });
        deltaLoop.setFps(300);
        deltaLoop.start();
    }

    int FileBoot::boot(int &argc, char **&argv) {
        int n_argc = 1;
        const auto app = Gtk::Application::create(
                n_argc,
                argv,
                "dev.tindersamurai.xmotion"
        );
        prepare_ocv();
        prepare_gui();
        prepare_loop();
        return app->run(*window);
    }

} // xm