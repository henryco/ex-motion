//
// Created by henryco on 11/25/23.
//

#ifndef STEREOX_GTK_CAM_PARAMS_H
#define STEREOX_GTK_CAM_PARAMS_H

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <gtkmm/box.h>
#include <gtkmm/scrolledwindow.h>

namespace eox::xgtk {

    typedef struct {
        uint id;
        std::string name;
        int min;
        int max;
        int step;
        int default_value;
        int value;
    } GtkCamProp;

    class GtkCamParams : public Gtk::ScrolledWindow {
    private:
        static inline const auto log =
                spdlog::stdout_color_mt("gtk_cam_params");

        std::function<int(uint, int)> onUpdateCallback = [](uint a, int b) { return b; };
        std::function<void()> onResetCallback = []() {};
        std::function<void()> onSaveCallback = []() {};

        sigc::connection debounce_connection;
        bool programmatic_change = false;

    public:
        static inline const uint DELAY_MS = 500;

        GtkCamParams();

        ~GtkCamParams() override = default;

        void onUpdate(std::function<int(uint, int)> callback);

        void onReset(std::function<void()> callback);

        void onSave(std::function<void()> callback);

        void setProperties(const std::vector<GtkCamProp> &properties);
    };

} // eox

#endif //STEREOX_GTK_CAM_PARAMS_H
