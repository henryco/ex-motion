//
// Created by henryco on 4/20/24.
//

#ifndef XMOTION_CAM_PARAMS_WINDOW_H
#define XMOTION_CAM_PARAMS_WINDOW_H

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <gtkmm/window.h>

#include "gtk_cam_params.h"

namespace xm {

    class CamParamsWindow : public Gtk::Window {

        static inline const auto log =
                spdlog::stdout_color_mt("cam_params_window");

    protected:
        eox::xgtk::GtkCamParams camParams;

    public:

    };

} // xm

#endif //XMOTION_CAM_PARAMS_WINDOW_H
