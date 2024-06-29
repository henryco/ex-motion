//
// Created by henryco on 28/06/24.
//

#ifndef XMOTION_JSON_CONFIG_GUI_H
#define XMOTION_JSON_CONFIG_GUI_H

#include <vector>
namespace xm::data {

    typedef struct {
        /**
         * Gui window width
         */
        int w;

        /**
         * Gui window height
         */
        int h;
    } GuiFrame;

    typedef struct {
        /**
         * Layout of gui grid [2]: {rows, columns}
         */
        std::vector<int> layout;

        GuiFrame frame;

        /**
         * Force vertical layout (optional)
         */
        bool vertical;

        /**
         * Gui window scaling
         */
        float scale;

        /**
         * Gui render FPS limit
         */
        int fps;
    } Gui;

    typedef struct {
        /**
         * Use dummy source of frames (tests only)
         */
        bool capture_dummy;

        /**
         * Use faster method of frames retrieval for camera devices
         * (not recommended)
         */
        bool capture_fast;

        /**
         * Debug mode
         */
        bool debug;

        /**
         * Default numbers of cpu cores available
         */
        int cpu;
    } Misc;

}

#endif //XMOTION_JSON_CONFIG_GUI_H
