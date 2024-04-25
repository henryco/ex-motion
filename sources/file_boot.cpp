//
// Created by henryco on 4/16/24.
//

#include <opencv2/opencv.hpp>
#include <gtkmm/application.h>

#include "../xmotion/boot/file_boot.h"
#include "../xmotion/utils/eox_globals.h"

namespace xm {

    void FileBoot::open_project(const char *argv) {
        project_file = xm::data::prepare_project_file(argv);
        config = xm::data::config_from_file(project_file);

        eox::globals::THREAD_POOL_CORES_MAX = config.misc.cpu;
    }

    int FileBoot::boostrap(int &argc, char **&argv) {
        int n_argc = 1;
        const auto app = Gtk::Application::create(
                n_argc,
                argv,
                "dev.tindersamurai.xmotion"
        );
        prepare_logic();
        prepare_cam();
        prepare_gui();
        start_loop(300);
        return app->run(*window);
    }

} // xm