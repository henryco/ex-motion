//
// Created by henryco on 11/21/23.
//

#ifndef STEREOX_GL_IMAGE_H
#define STEREOX_GL_IMAGE_H

#include <gtkmm/box.h>
#include <gtkmm/label.h>
#include <gtkmm/glarea.h>
#include <opencv2/core/mat.hpp>
#include "../../core/ogl/texture_1.h"
#include "../../core/ocl/ocl_data.h"
#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace eox::xgtk {

class GLImage : public Gtk::Box {
private:

    static inline const auto log =
            spdlog::stdout_color_mt("gl_image");

    std::function<void(int num, int x, int y, bool released)> callback = [](int a, int b, int c, bool d){};

    std::vector<std::unique_ptr<Gtk::Widget>> widgets;
    std::vector<std::unique_ptr<xogl::Texture1>> textures;
    std::vector<std::unique_ptr<Gtk::GLArea>> glAreas;
    std::vector<xm::ocl::Image2D> cl_frames;
    std::vector<cv::UMat> u_frames;
    std::vector<cv::Mat> frames;
    std::vector<bool> initialized;

    GLenum format = GL_RGB;
    int width = 0;
    int height = 0;

    int v_w = 0;
    int v_h = 0;

    size_t rows = 0;
    size_t cols = 0;

protected:
    std::function<bool(const Glib::RefPtr<Gdk::GLContext> &)> renderFunc(size_t num);
    std::function<void()> initFunc(size_t num);
    cv::Mat fitSize(const cv::Mat &in) const;
    cv::UMat fitSize(const cv::UMat &in) const;

public:
    GLImage() = default;
    ~GLImage() override;

    void init(size_t number, int width, int height, std::vector<std::string> ids, GLenum format = GL_RGB);
    void init(size_t number, int width, int height, GLenum format = GL_RGB);

    void init(size_t rows, size_t cols, size_t number, int width, int height, std::vector<std::string> ids, GLenum format = GL_RGB);
    void init(size_t rows, size_t cols, size_t number, int width, int height, GLenum format = GL_RGB);

    void setFrames(const std::vector<cv::Mat>& _frames);

    void setFrames(const std::vector<cv::UMat>& _frames);

    void setFrames(const std::vector<xm::ocl::Image2D>& _frames);

    void update();

    void scale(float _scale);
    
    void resize(int width = -1, int height = -1);

    int getViewWidth() const;

    int getViewHeight() const;

    void setMouseCallback(std::function<void(int num, int x, int y, bool released)> callback);
};

} // xgtk

#endif //STEREOX_GL_IMAGE_H
