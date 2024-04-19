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
        const auto frames = camera.capture();
        window->setFps((int) fps);
        window->refresh(frames);
    }

    void FileBoot::open_project(const char *argv) {
        project_path = xm::data::prepare_project_file(argv);
        config = xm::data::config_from_file(project_path);

        log->info("type: {}", config.type);
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
        button->proxy().signal_clicked().connect([this]() {
            // TODO: camera config
        });

        window = std::make_unique<xm::SimpleImageWindow>();
        window->init(config.camera.capture[0].width, config.camera.capture[0].height, config.camera._names);
        window->scale(config.gui.scale);
        window->add_one(*button);
        window->show_all_children();
    }

    void FileBoot::prepare_cam() {
        const auto project_dir = xm::data::prepare_project_dir(project_path);
        camera.setFastMode(config.camera.fast);
        for (const auto &c: config.camera.capture) {
            camera.open(c.id, c.codec, c.width, c.height, c.fps, c.buffer);

            std::filesystem::path file = c.name + ".xcam";
            std::filesystem::path conf = project_dir / file;

            if (!std::filesystem::exists(conf)) {
                log->debug("No configuration file found for camera device: {} | {} ", c.id, c.name);
                continue;
            }

            std::ifstream is(conf);
            log->debug("reading configuration for camera device: {} | {}", c.id, c.name);
            camera.read(is, c.id, c.name);
            is.close();
        }
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
        prepare_cam();
        prepare_loop();
        return app->run(*window);
    }

} // xm