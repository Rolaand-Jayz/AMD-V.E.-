#pragma once

// ─────────────────────────────────────────────────────────────────
// filter_catalog.hpp — Unified embedded filter/shader library
//
// Every filter ships inside the binary as an embedded string —
// either a GLSL shader (mpv / libplacebo format) or a VapourSynth
// script fragment.  Users toggle filters on/off and tweak their
// exposed parameters; the backend stitches selected filters into
// the processing pipeline.
// ─────────────────────────────────────────────────────────────────

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ave/types.hpp"

namespace ave {

// ─── Filter runtime target ──────────────────────────────────────
enum class FilterRuntime {
    Glsl,          // mpv / libplacebo GLSL hook shader
    VapourSynth    // VapourSynth .vpy script fragment
};

// ─── Filter category (for UI grouping) ──────────────────────────
enum class FilterCategory {
    Upscale,
    Sharpen,
    Denoise,
    Deblur,
    Dehalo,
    ColorCorrection,
    Restoration,
    LineArt,
    Grain,
    Interpolation,
    Utility
};

std::string toString(FilterCategory cat);
std::string toString(FilterRuntime rt);

// ─── Parameter descriptor ───────────────────────────────────────
struct FilterParamDesc {
    std::string key;          // e.g. "strength"
    std::string label;        // e.g. "Strength"
    std::string description;  // tooltip
    double      minVal   = 0.0;
    double      maxVal   = 1.0;
    double      defVal   = 0.5;
    double      step     = 0.01;
    bool        isInt    = false;   // render as int spinner
};

// ─── Single embedded filter ─────────────────────────────────────
struct EmbeddedFilter {
    std::string      id;           // unique key, e.g. "glsl.cas"
    std::string      name;         // "Contrast Adaptive Sharpening"
    std::string      description;  // one-liner
    FilterCategory   category;
    FilterRuntime    runtime;
    StageKind        stageKind;    // which pipeline stage it belongs to
    int              sortOrder;    // within its category (lower = earlier)
    std::string      source;       // full shader / script source
    std::vector<FilterParamDesc> params;
};

// ─── Active filter instance (user selection) ─────────────────────
struct ActiveFilter {
    std::string id;           // references EmbeddedFilter::id
    bool        enabled = true;
    std::unordered_map<std::string, double> paramValues;  // overrides
};

// ─── Catalog API ─────────────────────────────────────────────────

// Return all embedded filters (populated once at startup).
const std::vector<EmbeddedFilter>& allEmbeddedFilters();

// Lookup by id (returns nullptr if not found).
const EmbeddedFilter* findFilter(const std::string& id);

// Get all filters for a given category.
std::vector<const EmbeddedFilter*> filtersForCategory(FilterCategory cat);

// Get all filters for a given runtime.
std::vector<const EmbeddedFilter*> filtersForRuntime(FilterRuntime rt);

// Get all filters for a given stage kind.
std::vector<const EmbeddedFilter*> filtersForStage(StageKind kind);

// Build a concise display label for an enabled filter.
std::string displayNameForFilter(const ActiveFilter& filter);

// Resolve a shader source with parameter substitution.
// Placeholders like {{STRENGTH}} are replaced by paramValues.
std::string resolveSource(const EmbeddedFilter& filter,
                          const std::unordered_map<std::string, double>& paramValues);

}  // namespace ave
