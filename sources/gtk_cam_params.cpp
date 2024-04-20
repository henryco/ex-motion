//
// Created by henryco on 11/25/23.
//

#include <gtkmm/label.h>
#include <gtkmm/cssprovider.h>
#include <gtkmm/scale.h>
#include <gtkmm/spinbutton.h>
#include <glibmm/main.h>
#include <gtkmm/button.h>

#include "../xmotion/gtk/gtk_cam_params.h"
#include "../xmotion/gtk/gtk_utils.h"

namespace eox::xgtk {

    GtkCamParams::GtkCamParams() {
        set_policy(Gtk::POLICY_NEVER, Gtk::POLICY_AUTOMATIC);
    }

    void GtkCamParams::onUpdate(std::function<int(uint, int)> callback) {
        this->onUpdateCallback = std::move(callback);
    }

    void GtkCamParams::onReset(std::function<void()> callback) {
        this->onResetCallback = std::move(callback);
    }

    void GtkCamParams::onSave(std::function<void()> callback) {
        this->onSaveCallback = std::move(callback);
    }

#pragma clang diagnostic push
#pragma ide diagnostic ignored "ConstantConditionsOC"
#pragma ide diagnostic ignored "UnusedValue"
#pragma ide diagnostic ignored "UnreachableCode"

    void GtkCamParams::setProperties(const std::vector<GtkCamProp>& properties) {
        remove();

        auto v_box = Gtk::make_managed<Gtk::Box>(Gtk::ORIENTATION_VERTICAL);
        auto button_box = Gtk::make_managed<Gtk::Box>(Gtk::ORIENTATION_HORIZONTAL);
        v_box->pack_start(*button_box, Gtk::PACK_SHRINK);

        auto reset = Gtk::make_managed<Gtk::Button>();
        reset->signal_clicked().connect([this]() { onResetCallback(); });
        reset->get_style_context()->add_class("button-reset");
        reset->set_size_request(-1, 30);
        reset->set_label("Hardware Defaults");
        eox::xgtk::add_style(*reset, R"css(
            .button-reset {
                 margin-right: 5px;
                 margin-bottom: 5px;
             }
        )css");
        button_box->pack_start(*reset, Gtk::PACK_SHRINK);

        auto save = Gtk::make_managed<Gtk::Button>();
        save->signal_clicked().connect([this]() { onSaveCallback(); });
        save->get_style_context()->add_class("button-save");
        save->set_size_request(-1, 30);
        save->set_label("Save settings");
        eox::xgtk::add_style(*save, R"css(
            .button-save {
                 margin-right: 5px;
                 margin-bottom: 5px;
             }
        )css");
        button_box->pack_start(*save, Gtk::PACK_SHRINK);

        for (const auto &prop: properties) {
            auto c_box = Gtk::make_managed<Gtk::Box>(Gtk::ORIENTATION_VERTICAL);
            auto h_box = Gtk::make_managed<Gtk::Box>(Gtk::ORIENTATION_HORIZONTAL);
            auto scale = Gtk::make_managed<Gtk::Scale>(Gtk::ORIENTATION_HORIZONTAL);
            auto entry = Gtk::make_managed<Gtk::SpinButton>();
            auto label = Gtk::make_managed<Gtk::Label>();

            {
                // Control container
                c_box->set_size_request(300, 100);
                c_box->get_style_context()->add_class("cam-param-c_box");
                eox::xgtk::add_style(*c_box, R"css(
                    .cam-param-c_box {
                        background-color: white;
                        border-bottom: 1px solid lightgrey;
                        padding-right: 20px;
                        padding-left: 15px;
                        padding-top: 10px;
                    }
                )css");

                {
                    // Label
                    label->set_text(prop.name);
                    label->set_alignment(0);
                    label->get_style_context()->add_class("cam-param-label");
                    eox::xgtk::add_style(*label, R"css(
                        .cam-param-label {
                            padding-bottom: 10px;
                        }
                    )css");
                    c_box->pack_start(*label, Gtk::PACK_SHRINK);
                }

                {
                    // Slider
                    scale->set_size_request(260);
                    scale->set_range(prop.min, prop.max);
                    scale->set_increments(prop.step, std::min(prop.max, prop.step * 10));
                    scale->set_digits(0);
                    scale->set_value_pos(Gtk::POS_TOP);
                    scale->set_value(prop.value);
                    scale->signal_value_changed().connect([s_ptr = &*scale, e_ptr = &*entry, prop, this]() {
                        if (programmatic_change)
                            return;
                        programmatic_change = true;
                        {
                            auto value = (int) s_ptr->get_value();
                            const auto corr = value - (prop.min % prop.step);
                            const auto reminder = corr % prop.step;
                            if (reminder != 0) {
                                value = value + prop.step - reminder;
                                s_ptr->set_value(value);
                            }

                            e_ptr->set_value(value);

                            if (debounce_connection.connected())
                                debounce_connection.disconnect();

                            debounce_connection = Glib::signal_timeout().connect([this, prop, value, e_ptr, s_ptr]() {
                                programmatic_change = true;
                                {
                                    const auto result = onUpdateCallback(prop.id, value);
                                    if (value != result)
                                        s_ptr->set_value(result);
                                    e_ptr->set_value(result);
                                }
                                programmatic_change = false;
                                return false;
                            }, DELAY_MS);
                        }
                        programmatic_change = false;
                    });

                    h_box->pack_start(*scale, Gtk::PACK_SHRINK);
                }

                {
                    // Spin button
                    entry->set_numeric(true);
                    entry->set_snap_to_ticks(true);
                    entry->set_update_policy(Gtk::UPDATE_IF_VALID);
                    entry->set_size_request(50, 50);
                    entry->set_range(prop.min, prop.max);
                    entry->set_increments(prop.step, std::min(prop.max, prop.step * 10));
                    entry->set_digits(0);
                    entry->set_width_chars(5);
                    entry->set_value(prop.value);
                    entry->signal_value_changed().connect([s_ptr = &*scale, e_ptr = &*entry, prop, this]() {
                        if (programmatic_change)
                            return;
                        programmatic_change = true;
                        {
                            auto value = (int) e_ptr->get_value();
                            const auto corr = value - (prop.min % prop.step);
                            const auto reminder = corr % prop.step;
                            if (reminder != 0) {
                                value = value + prop.step - reminder;
                                e_ptr->set_value(value);
                            }
                            s_ptr->set_value(value);

                            if (debounce_connection.connected())
                                debounce_connection.disconnect();

                            debounce_connection = Glib::signal_timeout().connect([this, prop, value, e_ptr, s_ptr]() {
                                programmatic_change = true;
                                {
                                    const auto result = onUpdateCallback(prop.id, value);
                                    if (value != result)
                                        e_ptr->set_value(result);
                                    s_ptr->set_value(result);
                                }
                                programmatic_change = false;
                                return false;
                            }, DELAY_MS);
                        }
                        programmatic_change = false;
                    });

                    h_box->pack_end(*entry, Gtk::PACK_SHRINK);
                }
                c_box->pack_start(*h_box, Gtk::PACK_SHRINK);
            }
            v_box->pack_start(*c_box, Gtk::PACK_SHRINK);
        }
        add(*v_box);
    }

#pragma clang diagnostic pop

} // eox
