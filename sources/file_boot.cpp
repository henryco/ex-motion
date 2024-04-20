//
// Created by henryco on 4/16/24.
//

#include <filesystem>
#include <opencv2/opencv.hpp>
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

    void FileBoot::prepare_gui() {
        auto button_conf = Gtk::make_managed<xm::SmallButton>("c");
        button_conf->proxy().signal_clicked().connect([this]() {
            if (params_window->get_visible()) {
                params_window->hide();
                window->activate_focus();
                params_window->unset_focus();
            } else {
                params_window->show();
                window->unset_focus();
                params_window->activate_focus();
            }
        });

        auto button_start = Gtk::make_managed<xm::SmallButton>("s");
        button_start->proxy().signal_clicked().connect([this]() {
            // TODO: start
        });

        window = std::make_unique<xm::SimpleImageWindow>();
        window->init(config.camera.capture[0].width, config.camera.capture[0].height, config.camera._names);
        window->scale(config.gui.scale);
        window->add_one(*button_conf);
        window->add_one(*button_start);
        window->set_resizable(false);
        window->show_all_children();

        params_window = std::make_unique<xm::CamParamsWindow>();
        params_window->set_type_hint(Gdk::WINDOW_TYPE_HINT_DIALOG);
        params_window->set_visible(false);
        // TODO
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

    int FileBoot::boostrap(int &argc, char **&argv) {
        int n_argc = 1;
        const auto app = Gtk::Application::create(
                n_argc,
                argv,
                "dev.tindersamurai.xmotion"
        );
        prepare_cam();
        prepare_gui();
        start_loop(300);
        return app->run(*window);
    }

} // xm