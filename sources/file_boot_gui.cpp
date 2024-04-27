//
// Created by henryco on 4/21/24.
//

#include "../xmotion/gtk/small_button.h"
#include "../xmotion/boot/file_boot.h"

namespace xm {

    void FileBoot::load_device_params() {
        for (const auto &cap: config.camera.capture) {
            const auto props = camera.getControls(cap.id);
            std::vector<eox::xgtk::GtkCamProp> vec;
            vec.reserve(props.controls.size());
            for (const auto &c: props.controls) {
                vec.push_back({
                                      .id = c.id,
                                      .name = c.name,
                                      .min = c.min,
                                      .max = c.max,
                                      .step = c.step,
                                      .default_value = c.default_value,
                                      .value = c.value
                              });
            }
            params_window->add_camera(cap.id, cap.name, vec);
        }
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

        auto button_start = Gtk::make_managed<xm::SmallButton>("S");
        button_start->proxy().signal_clicked().connect([this, button_start]() {
            if (logic->is_active()) logic->stop();
            else logic->start();
            button_start->proxy().set_label(logic->is_active() ? "(S)" : "S");
        });

        auto button_bypass = Gtk::make_managed<xm::SmallButton>("B");
        button_bypass->proxy().signal_clicked().connect([this, button_bypass]() {
            bypass = !bypass;
            button_bypass->proxy().set_label(bypass ? "(B)" : "B");
        });

        const auto &cam = config.camera;
        const auto &gui = config.gui;
        window = std::make_unique<xm::SimpleImageWindow>();

        if (gui.layout.size() < 2)
            window->init(cam.capture[0].region.w, cam.capture[0].region.h, cam._names, gui.vertical);
        else
            window->init(cam.capture[0].region.w, cam.capture[0].region.h, cam._names, gui.layout[0], gui.layout[1]);

        window->scale(config.gui.scale);
        window->add_one(*button_conf);
        window->add_one(*button_start);
        window->add_one(*button_bypass);
        window->set_resizable(false);
        window->show_all_children();
        window->signal_delete_event().connect([this](GdkEventAny *any_event) {
            stop_loop();
            return false;
        });

        params_window = std::make_unique<xm::CamParamsWindow>();
        params_window->set_type_hint(Gdk::WINDOW_TYPE_HINT_DIALOG);
        params_window->set_visible(false);
        params_window->set_size_request(-1, (int) (gui.scale * (float) cam.capture[0].region.h));

        params_window->onUpdate([this](const std::string &device_id, uint id, int value) -> int {
            return on_camera_update(device_id, id, value);
        });
        params_window->onReset([this](const std::string &device_id) {
            on_camera_reset(device_id);
            params_window->clear();
            load_device_params();
            params_window->show_all_children(true);
        });
        params_window->onSave([this](const std::string &device_id) {
            on_camera_save(device_id);
        });

        load_device_params();
        params_window->init();
    }

}