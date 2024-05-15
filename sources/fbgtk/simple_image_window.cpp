//
// Created by henryco on 4/17/24.
//

#include "../../xmotion/fbgtk/gtk/simple_image_window.h"

namespace xm {

    SimpleImageWindow::SimpleImageWindow() {
        dispatcher.connect(sigc::mem_fun(*this, &SimpleImageWindow::on_dispatcher_signal));
        signal_size_allocate().connect([this](const Gtk::Allocation& allocation) {
            onResize(allocation);
        });
    }

    void SimpleImageWindow::refresh(const std::vector<cv::Mat>& _frames) {
        glImage.setFrames(_frames);
        refresh(true);
    }

    void SimpleImageWindow::refresh(bool _redraw) {
        redraw = _redraw;
        dispatcher.emit();
    }

    void SimpleImageWindow::on_dispatcher_signal() {
        if (redraw)
            glImage.update();

        if (fps_changed && fps < 1000)
            set_title("approx. fps: " + std::to_string((int) fps));
        else if (fps_changed && fps >= 1000)
            set_title("approx. fps: >= 1000");

        fps_changed = false;
        redraw = false;
    }

    void SimpleImageWindow::onResize(const Gtk::Allocation &allocation) {
//        glImage.resize(allocation.get_width(), allocation.get_height());
        // TODO FIXME
    }

    void SimpleImageWindow::init(int width, int height, const std::vector<std::string>& ids, bool vertical) {
        if (vertical)
            init(width, height, ids, (int) ids.size(), 1);
        else
            init(width, height, ids, 1, (int) ids.size());
    }

    void SimpleImageWindow::init(int width, int height, const std::vector<std::string> &ids, int rows, int cols) {
        layout_h.set_orientation(Gtk::ORIENTATION_HORIZONTAL);
        layout_v.set_orientation(Gtk::ORIENTATION_VERTICAL);

        glImage.init(rows, cols, ids.size(), width, height, ids, GL_BGR);

        layout_h.pack_start(glImage, Gtk::PACK_EXPAND_WIDGET);
        layout_h.pack_start(layout_v, Gtk::PACK_SHRINK);
        add(layout_h);

        show_all_children();
    }

    void SimpleImageWindow::add_one(Gtk::Widget &widget, Gtk::PackOptions packOptions) {
        layout_v.pack_start(widget, packOptions);
    }

    void SimpleImageWindow::scale(float scale) {
        glImage.scale(scale);
    }

    void SimpleImageWindow::setFps(int _fps) {
        const auto old = fps;
        fps = std::min(_fps, 1000);
        if (old != fps)
            fps_changed = true;
    }

} // xm