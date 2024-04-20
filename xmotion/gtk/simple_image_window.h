//
// Created by henryco on 4/17/24.
//

#ifndef XMOTION_SIMPLE_IMAGE_WINDOW_H
#define XMOTION_SIMPLE_IMAGE_WINDOW_H

#include "gl_image.h"
#include <gtkmm/window.h>
#include <glibmm/dispatcher.h>

namespace xm {

    class SimpleImageWindow : public Gtk::Window {

    private:
        Glib::Dispatcher dispatcher;
        eox::xgtk::GLImage glImage;
        Gtk::Box layout_h;
        Gtk::Box layout_v;

        bool fps_changed = false;
        int fps = 0;

    protected:
        void on_dispatcher_signal();

        void onResize(const Gtk::Allocation &allocation);

    public:
        SimpleImageWindow();

        void refresh(const std::vector<cv::Mat>& _frames);

        void refresh();

        void init(int width, int height, const std::vector<std::string>& ids);

        void add_one(Gtk::Widget &widget, Gtk::PackOptions packOptions = Gtk::PACK_SHRINK);

        void scale(float scale);

        void setFps(int fps);
    };

} // xm

#endif //XMOTION_SIMPLE_IMAGE_WINDOW_H
