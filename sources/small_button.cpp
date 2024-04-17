//
// Created by henryco on 4/18/24.
//

#include "../xmotion/gtk/small_button.h"
#include "../xmotion/gtk/gtk_utils.h"

namespace xm {
    Gtk::Button &SmallButton::proxy() {
        return button;
    }

    SmallButton::SmallButton(const std::string &label) {
        inner.set_orientation(Gtk::ORIENTATION_HORIZONTAL);
        inner.pack_start(button, Gtk::PACK_SHRINK);
        set_orientation(Gtk::ORIENTATION_VERTICAL);
        pack_start(inner, Gtk::PACK_SHRINK);
        button.set_label(label);
        button.get_style_context()->add_class("button-style");
        eox::xgtk::add_style(button, R"css(
            .button-style {
                        background-color: white;
                        border: 1px solid lightgrey;
                        padding: 2px;
            }
        )css");
    }
} // xm