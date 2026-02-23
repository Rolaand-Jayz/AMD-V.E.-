#include "ave/stage.hpp"

#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace ave {

std::string parameterValueToString(const ParameterValue& value) {
    if (std::holds_alternative<std::int64_t>(value)) {
        return std::to_string(std::get<std::int64_t>(value));
    }
    if (std::holds_alternative<double>(value)) {
        std::ostringstream os;
        os << std::setprecision(12) << std::get<double>(value);
        return os.str();
    }
    if (std::holds_alternative<bool>(value)) {
        return std::get<bool>(value) ? "true" : "false";
    }
    return std::get<std::string>(value);
}

bool tryGetDouble(const std::unordered_map<std::string, ParameterValue>& params,
                  const std::string& key,
                  double& outValue) {
    const auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }

    if (const auto* val = std::get_if<double>(&it->second)) {
        outValue = *val;
        return true;
    }

    if (const auto* val = std::get_if<std::int64_t>(&it->second)) {
        outValue = static_cast<double>(*val);
        return true;
    }

    if (const auto* val = std::get_if<std::string>(&it->second)) {
        char* end = nullptr;
        const double parsed = std::strtod(val->c_str(), &end);
        if (end != val->c_str() && *end == '\0') {
            outValue = parsed;
            return true;
        }
    }

    return false;
}

bool tryGetInt(const std::unordered_map<std::string, ParameterValue>& params,
               const std::string& key,
               std::int64_t& outValue) {
    const auto it = params.find(key);
    if (it == params.end()) {
        return false;
    }

    if (const auto* val = std::get_if<std::int64_t>(&it->second)) {
        outValue = *val;
        return true;
    }

    if (const auto* val = std::get_if<double>(&it->second)) {
        outValue = static_cast<std::int64_t>(*val);
        return true;
    }

    if (const auto* val = std::get_if<std::string>(&it->second)) {
        char* end = nullptr;
        const long long parsed = std::strtoll(val->c_str(), &end, 10);
        if (end != val->c_str() && *end == '\0') {
            outValue = static_cast<std::int64_t>(parsed);
            return true;
        }
    }

    return false;
}

}  // namespace ave
