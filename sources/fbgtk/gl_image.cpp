//
// Created by henryco on 11/21/23.
//

#include <iostream>
#include <utility>
#include <gtkmm/eventbox.h>
#include <opencv2/imgproc.hpp>
#include "../../xmotion/fbgtk/gtk/gl_image.h"

namespace eox::xgtk {

    GLImage::~GLImage() {
        textures.clear();
        glAreas.clear();
        frames.clear();
    }

    std::function<bool(const Glib::RefPtr<Gdk::GLContext> &)> GLImage::renderFunc(size_t num) {
        log->debug("create gl_render func: {}", num);
        return [num, this](const Glib::RefPtr<Gdk::GLContext>& context) -> bool {

            if (initialized.empty() || !initialized[num]) {
                log->warn("source [{}] is not initialized yet!", num);
                return true;
            }

            auto& texture = textures[num];
            if (texture == nullptr) {
                return false;
            }

            if (frames.empty()) {
                return true;
            }

            glClearColor(.0f, .0f, .0f, .0f);
            glClear(GL_COLOR_BUFFER_BIT);

            texture->render(xogl::Image(frames[num].data, width, height, format));

            return true;
        };
    }

    std::function<void()> GLImage::initFunc(size_t num) {
        log->debug("create gl_init func: {}", num);
        return [num, this]() {
            auto& area = glAreas[num];
            area->make_current();
            area->throw_if_error();
            textures[num]->init();
            initialized[num] = true;
        };
    }

    void GLImage::update() {
        for (auto &area: glAreas) {
            area->queue_render();
        }
    }

    void GLImage::setFrame(const cv::Mat &_frame) {
        frames.clear();
        frames.emplace_back(_frame);
    }

    void GLImage::setFrames(const std::vector<cv::Mat>& _frames) {
        frames.clear();
        for (const auto& item: _frames) {
            // YES, we NEED TO COPY IT
            frames.push_back(std::move(fitSize(item)));
        }
    }
    void GLImage::update(const std::vector<cv::Mat>& _frames) {
        setFrames(_frames);
        update();
    }

    void GLImage::init(size_t number, int _width, int _height, std::vector<std::string> ids, GLenum _format) {
        init(1, number, number, _width, _height, std::move(ids), _format);
    }

    void GLImage::init(size_t number, int _width, int _height, GLenum _format) {
        init(1, number, number, _width, _height, _format);
    }

    void GLImage::init(size_t _rows, size_t _cols, size_t _number, int _width, int _height, GLenum _format) {
        std::vector<std::string> _ids;
        _ids.reserve(_number);
        for (int i = 0; i < _number; i++) {
            _ids.push_back(std::to_string(i));
        }
        init(_rows, _cols, _number, _width, _height, _ids, _format);
    }

    void GLImage::init(size_t _rows, size_t _cols, size_t _number, int _width, int _height, std::vector<std::string> _ids, GLenum _format) {
        this->width = _width;
        this->height = _height;
        this->format = _format;

        initialized.clear();
        initialized.reserve(_number);

        textures.clear();
        textures.reserve(_number);

        glAreas.clear();
        glAreas.reserve(_number);

        frames.clear();
        frames.reserve(_number);

        widgets.clear();
        for (const auto &item: this->get_children()) {
            this->remove(*item);
        }

        this->v_w = (int) _cols * _width;
        this->v_h = (int) _rows * _height;

        this->rows = _rows;
        this->cols = _cols;

        size_t index = 0;
        for (size_t i = 0; i < _rows; i++) {
            auto h_box = std::make_unique<Gtk::Box>(Gtk::ORIENTATION_HORIZONTAL);
            h_box->set_halign(Gtk::ALIGN_CENTER);

            for (size_t k = 0; k < _cols; k++) {

                if (index >= _number)
                    break;

                auto area = std::make_unique<Gtk::GLArea>();
                area->signal_realize().connect(initFunc(index), false);
                area->signal_render().connect(renderFunc(index), false);
                area->set_size_request(_width, _height);
                area->set_auto_render(true);

                auto event_box = std::make_unique<Gtk::EventBox>();
                event_box->add(*area);
                event_box->signal_button_press_event().connect([this, p = area.get(), index](GdkEventButton *event) {
                    if (event->x < 0 || event->x >= p->get_width() || event->y < 0 || event->y >= p->get_height()) {
                        callback((int) index, 0, 0, true);
                        return true;
                    }
                    float xs = (float) p->get_width() / (float) width;
                    float ys = (float) p->get_height() / (float) height;
                    int x = (int) ((float) event->x / xs);
                    int y = (int)((float) event->y / ys);
                    callback((int) index, x, y, false);
                    return true;
                });
                event_box->signal_motion_notify_event().connect([this, p = area.get(), index](GdkEventMotion *event) {
                    if (event->x < 0 || event->x >= p->get_width() || event->y < 0 || event->y >= p->get_height()) {
                        callback((int) index, 0, 0, true);
                        return true;
                    }
                    float xs = (float) p->get_width() / (float) width;
                    float ys = (float) p->get_height() / (float) height;
                    int x = (int) ((float) event->x / xs);
                    int y = (int)((float) event->y / ys);
                    callback((int) index, x, y, false);
                    return true;
                });
                event_box->signal_button_release_event().connect([this, p = area.get(), index](GdkEventButton *event) {
                    if (event->x < 0 || event->x >= p->get_width() || event->y < 0 || event->y >= p->get_height()) {
                        callback((int) index, 0, 0, true);
                        return true;
                    }
                    float xs = (float) p->get_width() / (float) width;
                    float ys = (float) p->get_height() / (float) height;
                    int x = (int) ((float) event->x / xs);
                    int y = (int)((float) event->y / ys);
                    callback((int) index, x, y, true);
                    return true;
                });

                auto v_box = std::make_unique<Gtk::Box>(Gtk::ORIENTATION_VERTICAL);
                auto label = std::make_unique<Gtk::Label>();
                const std::string id = _ids.at(index);
                label->set_label(id);

                v_box->pack_start(*label, Gtk::PACK_SHRINK);
                v_box->pack_start(*event_box, Gtk::PACK_SHRINK);
                h_box->pack_start(*v_box, Gtk::PACK_SHRINK);

                widgets.push_back(std::move(label));
                widgets.push_back(std::move(v_box));
                widgets.push_back(std::move(event_box));
                widgets.push_back(std::move(v_box));
                glAreas.push_back(std::move(area));
                textures.push_back(std::make_unique<xogl::Texture1>());
                initialized.push_back(false);

                index += 1;
            }

            pack_start(*h_box, Gtk::PACK_SHRINK);
            widgets.push_back(std::move(h_box));
        }

        set_size_request(v_w, v_h);
        set_orientation(Gtk::ORIENTATION_VERTICAL);
        set_valign(Gtk::ALIGN_CENTER);
    }

    void GLImage::scale(float _scale) {
        resize((int) ((float) v_w * _scale), (int) ((float) v_h * _scale));
    }

    void GLImage::resize(int _width, int _height) {

        if (_width == -1 && _height == -1) {
            v_w = (int) cols * width;
            v_h = (int) rows * height;
        }

        else if (_width != -1 && _height == -1) {
            v_w = _width;

            const auto ratio = (float) (width * cols) / (float) (height * rows);
            v_h = (int) ((float) v_w / ratio);
        }

        else if (_width == -1) {
            v_h = _height;

            const auto ratio = (float) (width * cols) / (float) (height * rows);
            v_w = (int) ((float) v_h * ratio);
        }

        else {
            v_w = _width;
            v_h = _height;
        }

        for (const auto &area: glAreas) {
            area->set_size_request((int) (v_w / cols), (int) (v_h / rows));
        }
        set_size_request(v_w, v_h);
    }

    int GLImage::getViewWidth() const {
        return v_w;
    }

    int GLImage::getViewHeight() const {
        return v_h;
    }

    void GLImage::setMouseCallback(std::function<void(int, int, int, bool)> _callback) {
        this->callback = std::move(_callback);
    }

    cv::Mat GLImage::fitSize(const cv::Mat &in) const {
        if (in.cols == width && in.rows == height)
            return in;

        const float scale = std::min((float) width / (float) in.cols, (float) height / (float) in.rows);
        const float n_w = (float) in.cols * scale;
        const float n_h = (float) in.rows * scale;
        const float s_x = ((float) width - n_w) / 2.f;
        const float s_y = ((float) height - n_h) / 2.f;

        cv::Mat blob = cv::Mat::zeros(cv::Size(width, height), CV_8UC3);
        cv::Mat roi = blob(cv::Rect((int) s_x, (int) s_y, (int) n_w, (int) n_h));
        cv::resize(in, roi, cv::Size((int) n_w, (int) n_h),
                   0, 0, cv::INTER_NEAREST);
        return blob;
    }

} // xgtk