//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_GUI_H
#define XMOTION_JSON_CONFIG_GUI_H

#include <vector>
namespace xm::data {

    typedef struct {
        int w;
        int h;
    } GuiFrame;

    typedef struct {
        std::vector<int> layout;
        GuiFrame frame;
        bool vertical;
        float scale;
        int fps;
    } Gui;

    typedef struct {
        bool capture_dummy;
        bool capture_fast;
        bool debug;
        int cpu;
    } Misc;

}

#endif //XMOTION_JSON_CONFIG_GUI_H
