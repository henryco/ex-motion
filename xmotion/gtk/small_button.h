//
// Created by henryco on 4/18/24.
//

#ifndef XMOTION_SMALL_BUTTON_H
#define XMOTION_SMALL_BUTTON_H

#include <gtkmm/box.h>
#include <gtkmm/button.h>

namespace xm {

    class SmallButton : public Gtk::Box {
    private:
        Gtk::Box inner;
        Gtk::Button button;

    public:
        explicit SmallButton(const std::string &label = "");

        Gtk::Button& proxy();
    };

} // xm

#endif //XMOTION_SMALL_BUTTON_H
