#include "ave/scene_detector.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>

namespace ave {
namespace {

std::optional<std::string> runCapture(const std::string& cmd, int& exitCode) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        exitCode = -1;
        return std::nullopt;
    }
    std::string output;
    std::array<char, 1024> buf{};
    while (true) {
        const std::size_t n = std::fread(buf.data(), 1, buf.size(), pipe);
        if (n == 0) { break; }
        output.append(buf.data(), n);
    }
    exitCode = pclose(pipe);
    return output;
}

std::string quoteArg(const std::string& v) {
    std::string out = "\"";
    for (char ch : v) {
        if (ch == '\\' || ch == '"') { out.push_back('\\'); }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

// Parse ffprobe output: each matching line looks like
//   lavfi.scene_score=0.420300
//   pkt_pts_time=12.345670
// We accumulate frame records separated by blank lines.
std::vector<SceneCut> parseSceneOutput(const std::string& raw, double threshold) {
    std::vector<SceneCut> cuts;

    std::istringstream ss(raw);
    std::string line;

    double      curPts   = -1.0;
    double      curScore = -1.0;
    std::int64_t curFrame = 0;

    auto flush = [&]() {
        if (curScore >= threshold && curPts >= 0.0) {
            SceneCut cut;
            cut.frameIndex = curFrame;
            cut.pts        = curPts;
            cut.score      = curScore;
            cuts.push_back(cut);
        }
        curPts   = -1.0;
        curScore = -1.0;
    };

    while (std::getline(ss, line)) {
        if (line.empty()) {
            flush();
            ++curFrame;
            continue;
        }

        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) { continue; }

        const std::string key   = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);

        if (key == "pkt_pts_time" || key == "best_effort_timestamp_time") {
            try { curPts = std::stod(value); } catch (...) {}
        } else if (key == "lavfi.scene_score" || key == "tag:lavfi.scene_score") {
            try { curScore = std::stod(value); } catch (...) {}
        }
    }

    flush(); // handle last record if no trailing newline
    return cuts;
}

}  // namespace

bool SceneDetector::detect(const std::string&           inputPath,
                            const SceneDetectorOptions&  options,
                            std::vector<SceneCut>&       cuts,
                            std::string&                 error) const {
    // Build ffprobe command
    std::ostringstream cmd;
    cmd << "ffprobe -v error"
        << " -select_streams v:0"
        << " -show_entries frame=best_effort_timestamp_time,pkt_pts_time,tags"
        << " -of default=noprint_wrappers=1:nokey=0"
        << " -vf ";

    std::ostringstream filterStr;
    filterStr << "select=gte(scene\\," << std::fixed << std::setprecision(4) << options.threshold
              << "),metadata=print";
    cmd << quoteArg(filterStr.str());

    if (options.maxDurationSeconds > 0.0) {
        cmd << " -t " << std::fixed << std::setprecision(3) << options.maxDurationSeconds;
    }

    cmd << " " << quoteArg(inputPath);

    int exitCode = 0;
    const auto output = runCapture(cmd.str(), exitCode);

    if (!output.has_value()) {
        error = "Failed to run ffprobe for scene detection.";
        return false;
    }

    if (exitCode != 0) {
        error = "ffprobe exited with code " + std::to_string(exitCode) + " during scene detection.";
        return false;
    }

    cuts = parseSceneOutput(*output, options.threshold);
    return true;
}

bool SceneDetector::isCutAdjacent(const std::vector<SceneCut>& cuts,
                                   std::int64_t frameIndex,
                                   std::int64_t margin) {
    for (const auto& cut : cuts) {
        if (std::abs(cut.frameIndex - frameIndex) <= margin) {
            return true;
        }
    }
    return false;
}

}  // namespace ave
