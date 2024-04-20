//
// Created by henryco on 4/20/24.
//

#include "../xmotion/gtk/cam_params_window.h"

namespace xm {
    void CamParamsWindow::add_camera(const std::string &name, const std::vector<eox::xgtk::GtkCamProp>& props) {
        auto cam_params = Gtk::make_managed<eox::xgtk::GtkCamParams>();
        cam_params->setProperties(props);
        config_stack.add(*cam_params, name);
    }

    void CamParamsWindow::init() {
        layout_v.set_orientation(Gtk::ORIENTATION_VERTICAL);
        layout_v.pack_end(config_stack, Gtk::PACK_EXPAND_WIDGET);
        add(layout_v);
        show_all_children(true);
    }
} // xm