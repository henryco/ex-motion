//
// Created by henryco on 5/26/24.
//

#ifndef XMOTION_COMPOSE_H
#define XMOTION_COMPOSE_H

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

class EpiComposer {
    static inline const auto log =
            spdlog::stdout_color_mt("epi_composer");

    // TODO: Mergin two pairs into one
};

#endif //XMOTION_COMPOSE_H
