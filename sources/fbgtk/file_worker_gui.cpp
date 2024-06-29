//
// Created by henryco on 4/21/24.
//

#include "../../xmotion/fbgtk/file_worker.h"
#include "../../xmotion/fbgtk/gtk/small_button.h"

namespace xm {

    void FileWorker::update_gui(float fps) {
        Glib::signal_idle().connect([this, fps]() -> bool {
            if (window == nullptr)
                return false;
            window->setFps((int) fps);
            if (!bypass)
                window->refresh(logic->frames());
            return false;
        });
    }

    void FileWorker::load_device_params() {
        for (const auto &cap: config.captures) {
            const auto props = camera->getControls(cap.id);
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

    void FileWorker::prepare_gui() {
        Glib::signal_idle().connect([this]() -> bool {
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

            auto button_filter = Gtk::make_managed<xm::SmallButton>("F");
            button_filter->proxy().signal_clicked().connect([this, button_filter]() {
                do_filter = !do_filter;
                button_filter->proxy().set_label(do_filter ? "(F)" : "F");
            });

            const auto &captures = config.captures;
            const auto &gui = config.gui;

            std::vector<std::string> c_names;
            c_names.reserve(captures.size());
            for (const auto &d: captures)
                c_names.push_back(d.name);


            int fw, fh;
            if (gui.frame.w <= 0 || gui.frame.h <= 0) {
                fw = captures[0].region.w;
                fh = captures[0].region.h;
            } else {
                fw = gui.frame.w;
                fh = gui.frame.h;
            }

            if (gui.layout.size() < 2)
                window->init(fw, fh, c_names, gui.vertical);
            else
                window->init(fw, fh, c_names, gui.layout[0], gui.layout[1]);

            window->scale(config.gui.scale);
            window->add_one(*button_conf);
            window->add_one(*button_start);
            window->add_one(*button_bypass);
            window->add_one(*button_filter);
            window->set_resizable(false);
            window->show_all_children();

            params_window->set_type_hint(Gdk::WINDOW_TYPE_HINT_DIALOG);
            params_window->set_visible(false);

            const auto a1 = (int) (gui.scale * (float) captures[0].region.h);
            const auto a2 = window->get_screen()->get_height();
            const auto a3 = window->get_screen()->get_width();
            const auto a4 = std::min(a2, a3);
            const auto a5 = std::min(a1, a4);

            params_window->set_size_request(-1, std::min(a5, 500));

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

            return false;
        });
    }

}