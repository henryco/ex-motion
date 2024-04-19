//
// Created by henryco on 4/17/24.
//

#include "../xmotion/gtk/simple_image_window.h"

namespace xm {

    SimpleImageWindow::SimpleImageWindow() {
        dispatcher.connect(sigc::mem_fun(*this, &SimpleImageWindow::on_dispatcher_signal));
        signal_size_allocate().connect([this](const Gtk::Allocation& allocation) {
            onResize(allocation);
        });
    }

    void SimpleImageWindow::refresh(const std::vector<cv::Mat>& _frames) {
        glImage.setFrames(_frames);
        dispatcher.emit();
    }

    void SimpleImageWindow::refresh() {
        dispatcher.emit();
    }

    void SimpleImageWindow::on_dispatcher_signal() {
        glImage.update();
        if (fps_changed && fps < 1000)
            set_title("approx. fps: " + std::to_string((int) fps));
        else if (fps_changed && fps >= 1000)
            set_title("approx. fps: >= 1000");
        fps_changed = false;
    }

    void SimpleImageWindow::onResize(const Gtk::Allocation &allocation) {
        glImage.resize(allocation.get_width(), allocation.get_height());
    }

    void SimpleImageWindow::init(int width, int height, const std::vector<std::string>& ids) {
        glImage.init(ids.size(), width, height, ids);
        add_one(glImage);
        add(layout_h);
        show_all_children();
    }

    void SimpleImageWindow::add_one(Gtk::Widget &widget, Gtk::PackOptions packOptions) {
        layout_h.pack_start(widget, packOptions);
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