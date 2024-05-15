//
// Created by henryco on 4/20/24.
//

#include <utility>

#include "../../xmotion/fbgtk/gtk/cam_params_window.h"

namespace xm {
    void CamParamsWindow::add_camera(const std::string &id, const std::string &name, const std::vector<eox::xgtk::GtkCamProp>& props) {
        auto cam_params = Gtk::make_managed<eox::xgtk::GtkCamParams>();
        cam_params->setProperties(props);

        cam_params->onUpdate([this, id](uint prop_id, int value) {
            return on_update(id, prop_id, value);
        });

        cam_params->onSave([this, id]() {
            on_save(id);
        });

        cam_params->onReset([this, id]() {
            on_reset(id);
        });

        config_stack.add(*cam_params, name);
    }

    void CamParamsWindow::init() {
        layout_v.set_orientation(Gtk::ORIENTATION_VERTICAL);
        layout_v.pack_end(config_stack, Gtk::PACK_EXPAND_WIDGET);
        add(layout_v);
        show_all_children(true);
    }

    void CamParamsWindow::onUpdate(std::function<int(const std::string &, uint, int)> function) {
        on_update = std::move(function);
    }

    void CamParamsWindow::onReset(std::function<void(const std::string &)> function) {
        on_reset = std::move(function);
    }

    void CamParamsWindow::onSave(std::function<void(const std::string &)> function) {
        on_save = std::move(function);
    }

    void CamParamsWindow::clear() {
        config_stack.clear();
    }
} // xm