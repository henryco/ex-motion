//
// Created by henryco on 4/16/24.
//

#include <opencv2/opencv.hpp>
#include <gtkmm/application.h>

#include "../../xmotion/fbgtk/file_boot.h"
#include "../../xmotion/core/utils/eox_globals.h"
#include "../../xmotion/fbgtk/file_worker.h"

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

        window = new xm::SimpleImageWindow();
        params_window = new xm::CamParamsWindow();
        window->signal_delete_event().connect([this](GdkEventAny *any_event) {
            stop_loop();
            return false;
        });

        start_loop(999);
        return app->run(*window);
    }

    eox::util::DeltaWorker *FileBoot::worker() {
        return new xm::FileWorker(window, params_window, config, project_file);
    }

    FileBoot::~FileBoot() {
        delete params_window;
        delete window;
    }

} // xm