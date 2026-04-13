#include <atomic>
#include <csignal>
#include <iostream>

#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>
#endif

#include "ave/backend_manager.hpp"
#include "ave/cli.hpp"
#include "ave/telemetry.hpp"
#include "ave/video_processor.hpp"

namespace {
std::atomic<bool> g_cancelFlag{false};
volatile std::sig_atomic_t g_signalCount = 0;

void signalHandler(int /*sig*/) {
    if (g_signalCount > 0) {
        // Second Ctrl+C — force-exit immediately
        std::_Exit(130);
    }
    g_signalCount = 1;
    g_cancelFlag.store(true, std::memory_order_relaxed);
#if defined(__unix__) || defined(__APPLE__)
    static constexpr char kCancelMessage[] =
        "\nCancelling... (press Ctrl+C again to force quit)\n";
    (void)::write(STDERR_FILENO, kCancelMessage, sizeof(kCancelMessage) - 1);
#endif
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
