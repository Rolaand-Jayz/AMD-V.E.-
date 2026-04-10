#include <atomic>
#include <csignal>
#include <iostream>

#include "ave/backend_manager.hpp"
#include "ave/cli.hpp"
#include "ave/telemetry.hpp"
#include "ave/video_processor.hpp"

namespace {
std::atomic<bool> g_cancelFlag{false};

void signalHandler(int /*sig*/) {
    if (g_cancelFlag.load(std::memory_order_relaxed)) {
        // Second Ctrl+C — force-exit immediately
        std::_Exit(130);
    }
    g_cancelFlag.store(true, std::memory_order_relaxed);
    std::cerr << "\nCancelling… (press Ctrl+C again to force quit)\n";
}
}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    const ave::CliResult cli = ave::parseCli(argc, argv);
    if (cli.shouldExit) {
        return cli.exitCode;
    }

    if (cli.listBackends) {
        ave::BackendManager manager;
        const auto backends = manager.probeBackends();
        const auto diagnostics = manager.runtimeDiagnostics();
        const auto telemetryProbe = ave::probeAmdTelemetrySupport();
        std::cout << "AMD backend probe:" << std::endl;
        for (const auto& backend : backends) {
            std::cout << "- " << backend.name << ": " << (backend.available ? "available" : "unavailable")
                      << " (" << backend.detail << ")" << std::endl;
        }
        std::cout << "\nAMD runtime diagnostics:" << std::endl;
        std::cout << diagnostics.detailedText() << std::endl;
        std::cout << "\nAMD telemetry:" << std::endl;
        std::cout << telemetryProbe.summary() << std::endl;
        std::string telemetryError;
        if (const auto telemetry = ave::collectAmdTelemetry(telemetryError); telemetry.has_value()) {
            std::cout << telemetry->summary() << std::endl;
        } else if (!telemetryError.empty()) {
            std::cout << telemetryError << std::endl;
        }
        return 0;
    }

    if (!cli.job.has_value()) {
        std::cerr << "No job provided." << std::endl;
        return 1;
    }

    ave::VideoJob job = *cli.job;
    job.cancelFlag = &g_cancelFlag;

    ave::VideoProcessor processor;
    std::string error;
    if (!processor.process(job, error)) {
        if (g_cancelFlag.load()) {
            std::cerr << "Processing cancelled." << std::endl;
            return 130;
        }
        std::cerr << "error: " << error << std::endl;
        return 1;
    }

    std::cout << "done." << std::endl;
    return 0;
}
