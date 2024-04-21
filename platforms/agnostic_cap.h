//
// Created by henryco on 4/22/24.
//

#ifndef XMOTION_AGNOSTIC_CAP_H
#define XMOTION_AGNOSTIC_CAP_H

#include <string>
#include <vector>
#include <fstream>

namespace platform::cap {

    typedef struct {
        uint id;
        std::string name;
        int min;
        int max;
        int step;
        int default_value;
        int value;
    } camera_control;

    typedef struct {
        /**
         * device id
         */
        std::string id;

        /**
         * device controls
         */
        std::vector<camera_control> controls;

    } camera_controls;

    int video_capture_api();

    int index_from_id(const std::string &id);

    camera_controls query_controls(const std::string &id);

    void set_control_value(const std::string &device_id, uint prop_id, int value);

    void save(std::ostream &output_stream, const std::string &name, const camera_controls &control);

    camera_controls read(std::istream &input_stream, const std::string &name);
}


#endif //XMOTION_AGNOSTIC_CAP_H
