#include <iostream>

#include "ave/backend_manager.hpp"
#include "ave/cli.hpp"
#include "ave/video_processor.hpp"

int main(int argc, char** argv) {
    const ave::CliResult cli = ave::parseCli(argc, argv);
    if (cli.shouldExit) {
        return cli.exitCode;
    }

    if (cli.listBackends) {
        ave::BackendManager manager;
        const auto backends = manager.probeBackends();
        std::cout << "AMD backend probe:" << std::endl;
        for (const auto& backend : backends) {
            std::cout << "- " << backend.name << ": " << (backend.available ? "available" : "unavailable")
                      << " (" << backend.detail << ")" << std::endl;
        }
        return 0;
    }

    if (!cli.job.has_value()) {
        std::cerr << "No job provided." << std::endl;
        return 1;
    }

    ave::VideoProcessor processor;
    std::string error;
    if (!processor.process(*cli.job, error)) {
        std::cerr << "error: " << error << std::endl;
        return 1;
    }

    std::cout << "done." << std::endl;
    return 0;
}
