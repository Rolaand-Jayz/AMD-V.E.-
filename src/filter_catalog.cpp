// DISABLED: VapourSynth/GLSL/FilterCatalog feature — commented out, not removed.
#if 0  // ── entire file disabled ──────────────────────────────

// ─────────────────────────────────────────────────────────────────
// filter_catalog.cpp — All embedded GLSL shaders & VS presets
// ─────────────────────────────────────────────────────────────────
#include "ave/filter_catalog.hpp"

#include <algorithm>
#include <regex>
#include <sstream>
#include <unordered_map>

namespace ave {

// ═════════════════════════════════════════════════════════════════
//  Helpers
// ═════════════════════════════════════════════════════════════════

std::string toString(FilterCategory cat) {
    switch (cat) {
        case FilterCategory::Upscale:          return "Upscale";
        case FilterCategory::Sharpen:          return "Sharpen";
        case FilterCategory::Denoise:          return "Denoise";
        case FilterCategory::Deblur:           return "Deblur";
        case FilterCategory::Dehalo:           return "Dehalo";
        case FilterCategory::ColorCorrection:  return "Color Correction";
        case FilterCategory::Restoration:      return "Restoration";
        case FilterCategory::LineArt:          return "Line Art";
        case FilterCategory::Grain:            return "Grain";
        case FilterCategory::Interpolation:    return "Interpolation";
        case FilterCategory::Utility:          return "Utility";
    }
    return "Unknown";
}

std::string toString(FilterRuntime rt) {
    switch (rt) {
        case FilterRuntime::Glsl:        return "GLSL";
        case FilterRuntime::VapourSynth: return "VapourSynth";
    }
    return "Unknown";
}

// ═════════════════════════════════════════════════════════════════
//  GLSL Shader Sources (mpv / libplacebo hook format)
// ═════════════════════════════════════════════════════════════════

// ── Sharpening ──────────────────────────────────────────────────

static const char* const kGlslCAS = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Contrast Adaptive Sharpening (CAS)

vec4 hook() {
    vec2 ts = HOOKED_pt;
    vec4 c = HOOKED_texOff(vec2(0.0, 0.0));
    vec4 n = HOOKED_texOff(vec2(0.0, -1.0));
    vec4 s = HOOKED_texOff(vec2(0.0,  1.0));
    vec4 e = HOOKED_texOff(vec2( 1.0, 0.0));
    vec4 w = HOOKED_texOff(vec2(-1.0, 0.0));

    float mn = min(min(min(n.g, s.g), min(e.g, w.g)), c.g);
    float mx = max(max(max(n.g, s.g), max(e.g, w.g)), c.g);
    float amp = clamp(min(mn, 2.0 - mx) / mx, 0.0, 1.0);
    amp = sqrt(amp);

    float sharpness = {{SHARPNESS}};
    float peak = -1.0 / mix(8.0, 5.0, sharpness);
    float wt = amp * peak;
    vec3 result = (c.rgb + (n.rgb + s.rgb + e.rgb + w.rgb) * wt) / (1.0 + 4.0 * wt);
    return vec4(result, c.a);
}
)";

static const char* const kGlslAdaptiveSharpen = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Adaptive Sharpen

vec4 hook() {
    float curve = {{CURVE_HEIGHT}};
    vec2 pt = HOOKED_pt;
    // 3x3 neighbourhood
    vec3 a = HOOKED_texOff(vec2(-1,-1)).rgb;
    vec3 b = HOOKED_texOff(vec2( 0,-1)).rgb;
    vec3 cc = HOOKED_texOff(vec2( 1,-1)).rgb;
    vec3 d = HOOKED_texOff(vec2(-1, 0)).rgb;
    vec3 e = HOOKED_texOff(vec2( 0, 0)).rgb;
    vec3 f = HOOKED_texOff(vec2( 1, 0)).rgb;
    vec3 g = HOOKED_texOff(vec2(-1, 1)).rgb;
    vec3 h = HOOKED_texOff(vec2( 0, 1)).rgb;
    vec3 i = HOOKED_texOff(vec2( 1, 1)).rgb;

    // Sobel-like edge detection
    vec3 dx = (cc + 2.0*f + i) - (a + 2.0*d + g);
    vec3 dy = (g + 2.0*h + i) - (a + 2.0*b + cc);
    float edge = length(dx) + length(dy);
    edge = clamp(edge, 0.0, 1.0);

    // Unsharp masking with edge-adaptive falloff
    vec3 blur = (a + b + cc + d + e + f + g + h + i) / 9.0;
    vec3 detail = e - blur;
    float adapt = 1.0 - smoothstep(0.0, 0.7, edge);
    vec3 result = e + detail * curve * (0.5 + adapt * 0.5);
    return vec4(clamp(result, 0.0, 1.0), 1.0);
}
)";

static const char* const kGlslUnsharpMask = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Unsharp Mask

vec4 hook() {
    float amount = {{AMOUNT}};
    float radius_factor = {{RADIUS}};
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    // Gaussian-like blur via sampling neighbourhood
    vec3 blur = vec3(0.0);
    float total = 0.0;
    for (float x = -2.0; x <= 2.0; x += 1.0) {
        for (float y = -2.0; y <= 2.0; y += 1.0) {
            float w = exp(-(x*x + y*y) / (2.0 * radius_factor * radius_factor));
            blur += HOOKED_texOff(vec2(x, y)).rgb * w;
            total += w;
        }
    }
    blur /= total;
    vec3 sharp = c + (c - blur) * amount;
    return vec4(clamp(sharp, 0.0, 1.0), 1.0);
}
)";

static const char* const kGlslLumaSharpen = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Luma Sharpen

vec4 hook() {
    float sharp_str = {{STRENGTH}};
    float clamp_amt = {{CLAMP}};
    vec4 c = HOOKED_texOff(vec2(0, 0));
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float n = dot(HOOKED_texOff(vec2(0,-1)).rgb, vec3(0.2126,0.7152,0.0722));
    float s = dot(HOOKED_texOff(vec2(0, 1)).rgb, vec3(0.2126,0.7152,0.0722));
    float e = dot(HOOKED_texOff(vec2(1, 0)).rgb, vec3(0.2126,0.7152,0.0722));
    float w = dot(HOOKED_texOff(vec2(-1,0)).rgb, vec3(0.2126,0.7152,0.0722));
    float edge = luma * 4.0 - n - s - e - w;
    edge = clamp(edge * sharp_str, -clamp_amt, clamp_amt);
    return vec4(c.rgb + edge, c.a);
}
)";

// ── Denoising ───────────────────────────────────────────────────

static const char* const kGlslNlMeans = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Non-Local Means Denoise

vec4 hook() {
    float h = {{H_STRENGTH}};
    float h2 = h * h;
    int search = 3;
    int patch  = 1;
    vec3 center = HOOKED_texOff(vec2(0,0)).rgb;
    vec3 result = vec3(0.0);
    float wsum = 0.0;
    for (int sx = -search; sx <= search; sx++) {
        for (int sy = -search; sy <= search; sy++) {
            float dist = 0.0;
            for (int px = -patch; px <= patch; px++) {
                for (int py = -patch; py <= patch; py++) {
                    vec3 diff = HOOKED_texOff(vec2(float(px), float(py))).rgb
                              - HOOKED_texOff(vec2(float(sx+px), float(sy+py))).rgb;
                    dist += dot(diff, diff);
                }
            }
            float patchSize = float((2*patch+1)*(2*patch+1));
            dist /= patchSize;
            float w = exp(-dist / h2);
            result += HOOKED_texOff(vec2(float(sx), float(sy))).rgb * w;
            wsum += w;
        }
    }
    return vec4(result / wsum, 1.0);
}
)";

static const char* const kGlslBilateralDenoise = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Bilateral Denoise

vec4 hook() {
    float sigma_s = {{SPATIAL_SIGMA}};
    float sigma_r = {{RANGE_SIGMA}};
    vec3 center = HOOKED_texOff(vec2(0,0)).rgb;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    int r = int(ceil(sigma_s * 2.0));
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            vec3 s = HOOKED_texOff(vec2(float(x), float(y))).rgb;
            float sd = float(x*x + y*y);
            float sw = exp(-sd / (2.0*sigma_s*sigma_s));
            vec3 diff = s - center;
            float rw = exp(-dot(diff,diff) / (2.0*sigma_r*sigma_r));
            float w = sw * rw;
            acc += s * w;
            wsum += w;
        }
    }
    return vec4(acc / wsum, 1.0);
}
)";

static const char* const kGlslMedianDenoise = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Median 3x3 Denoise

#define s2(a,b) temp=a;a=min(a,b);b=max(temp,b);
#define mn3(a,b,c) s2(a,b);s2(a,c);
#define mx3(a,b,c) s2(b,c);s2(a,c);
#define mnmx3(a,b,c) mx3(a,b,c);s2(a,b);
#define mnmx4(a,b,c,d) s2(a,b);s2(c,d);s2(a,c);s2(b,d);
#define mnmx5(a,b,c,d,e) s2(a,b);s2(c,d);mn3(a,c,e);mx3(b,d,e);
#define mnmx6(a,b,c,d,e,f) s2(a,d);s2(b,e);s2(c,f);mn3(a,b,c);mx3(d,e,f);

vec4 hook() {
    float blend = {{BLEND}};
    vec3 v[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            v[i*3+j] = HOOKED_texOff(vec2(float(j-1), float(i-1))).rgb;
        }
    }
    vec3 temp;
    mnmx6(v[0],v[1],v[2],v[3],v[4],v[5]);
    mnmx5(v[1],v[2],v[3],v[4],v[6]);
    mnmx4(v[2],v[3],v[4],v[7]);
    mnmx3(v[3],v[4],v[8]);
    return vec4(mix(HOOKED_texOff(vec2(0,0)).rgb, v[4], blend), 1.0);
}
)";

static const char* const kGlslTemporalDenoise = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Temporal Averaging Denoise (spatial approx)

vec4 hook() {
    float strength = {{STRENGTH}};
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    // Approximate temporal denoise by averaging surrounding pixels
    vec3 avg = vec3(0.0);
    float count = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            avg += HOOKED_texOff(vec2(float(x),float(y))).rgb;
            count += 1.0;
        }
    }
    avg /= count;
    vec3 diff = c - avg;
    float d = length(diff);
    float w = smoothstep(0.0, strength * 0.1, d);
    return vec4(mix(avg, c, w), 1.0);
}
)";

// ── Debanding ───────────────────────────────────────────────────

static const char* const kGlslDeband = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Debanding (gradient smoothing)

vec4 hook() {
    float threshold = {{THRESHOLD}};
    float range_val = {{RANGE}};
    int grain = int({{GRAIN}});
    vec4 c = HOOKED_texOff(vec2(0,0));
    vec3 ref = c.rgb;
    vec3 avg = vec3(0.0);
    float count = 0.0;
    float r = range_val;
    // Sample in a diamond pattern
    for (float i = -r; i <= r; i += 1.0) {
        for (float j = -r; j <= r; j += 1.0) {
            if (abs(i) + abs(j) > r) continue;
            vec3 s = HOOKED_texOff(vec2(i, j)).rgb;
            vec3 diff = abs(s - ref);
            float maxdiff = max(max(diff.r, diff.g), diff.b);
            if (maxdiff < threshold / 255.0) {
                avg += s;
                count += 1.0;
            }
        }
    }
    if (count < 1.0) return c;
    avg /= count;
    // Optional slight dither to hide remaining bands
    float noise = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453);
    avg += (noise - 0.5) * float(grain) / 255.0;
    return vec4(avg, c.a);
}
)";

// ── Dehaloing ───────────────────────────────────────────────────

static const char* const kGlslDehalo = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Dehalo (edge-aware halo removal)

vec4 hook() {
    float strength = {{STRENGTH}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    // Dilate (max) and erode (min)
    vec3 mx = c.rgb, mn = c.rgb;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 s = HOOKED_texOff(vec2(float(x),float(y))).rgb;
            mx = max(mx, s);
            mn = min(mn, s);
        }
    }
    // Clamp center to local [min, max] of eroded/dilated
    vec3 clamped = clamp(c.rgb, mn, mx);
    // Weighted average of clamped and original
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float edge = length(mx - mn);
    float w = smoothstep(0.0, 0.3, edge) * strength;
    return vec4(mix(c.rgb, clamped, w), c.a);
}
)";

static const char* const kGlslFineEdgeDehalo = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Fine Edge Dehalo

vec4 hook() {
    float radius = {{RADIUS}};
    float darkstr = {{DARK_STRENGTH}};
    float brightstr = {{BRIGHT_STRENGTH}};
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    // Edge-aware spatial median approach
    vec3 minC = vec3(1.0), maxC = vec3(0.0);
    int r = int(radius);
    for (int x = -r; x <= r; x++) {
        for (int y = -r; y <= r; y++) {
            if (x == 0 && y == 0) continue;
            vec3 s = HOOKED_texOff(vec2(float(x),float(y))).rgb;
            minC = min(minC, s);
            maxC = max(maxC, s);
        }
    }
    vec3 overshoot = max(c - maxC, 0.0) * brightstr;
    vec3 undershoot = max(minC - c, 0.0) * darkstr;
    return vec4(c - overshoot + undershoot, 1.0);
}
)";

// ── Color Correction ─────────────────────────────────────────────

static const char* const kGlslColorBalance = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Color Balance

vec4 hook() {
    float lift_r = {{LIFT_R}};
    float lift_g = {{LIFT_G}};
    float lift_b = {{LIFT_B}};
    float gain_r = {{GAIN_R}};
    float gain_g = {{GAIN_G}};
    float gain_b = {{GAIN_B}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    vec3 lift = vec3(lift_r, lift_g, lift_b);
    vec3 gain = vec3(gain_r, gain_g, gain_b);
    c.rgb = (c.rgb - 0.5) * gain + 0.5 + lift;
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslVibrance = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Vibrance (intelligent saturation)

vec4 hook() {
    float vibrance = {{VIBRANCE}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float saturation = max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
    // Apply more boost to less-saturated pixels
    float boost = vibrance * (1.0 - saturation);
    c.rgb = mix(vec3(luma), c.rgb, 1.0 + boost);
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslAutoLevels = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Auto Levels (stretch histogram)

vec4 hook() {
    float black_point = {{BLACK_POINT}} / 255.0;
    float white_point = {{WHITE_POINT}} / 255.0;
    vec4 c = HOOKED_texOff(vec2(0,0));
    c.rgb = (c.rgb - black_point) / (white_point - black_point);
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslGammaCorrection = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Gamma Correction

vec4 hook() {
    float gamma = {{GAMMA}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    c.rgb = pow(max(c.rgb, 0.0), vec3(1.0 / gamma));
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslSaturationBoost = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Saturation Boost

vec4 hook() {
    float sat = {{SATURATION}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    c.rgb = mix(vec3(luma), c.rgb, sat);
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslContrastCurve = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC S-Curve Contrast

vec4 hook() {
    float contrast = {{CONTRAST}};
    float midpoint = {{MIDPOINT}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    // S-curve via smoothstep-like function
    c.rgb = mix(vec3(midpoint), c.rgb, contrast);
    // Apply sigmoid for smooth contrast
    c.rgb = 1.0 / (1.0 + exp(-6.0 * contrast * (c.rgb - midpoint))) ;
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslColorTemperature = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Color Temperature Shift

vec4 hook() {
    float temp = {{TEMPERATURE}};  // negative=cool, positive=warm
    float tint = {{TINT}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    // Warm shifts red up, blue down; cool does the opposite
    c.r += temp * 0.1;
    c.b -= temp * 0.1;
    c.g += tint * 0.1;
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

// ── Film Grain ──────────────────────────────────────────────────

static const char* const kGlslFilmGrain = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Film Grain (organic noise)

vec4 hook() {
    float intensity = {{INTENSITY}};
    float size = {{GRAIN_SIZE}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    // Hash-based pseudo-random noise
    vec2 uv = gl_FragCoord.xy / size;
    float n = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
    float n2 = fract(sin(dot(uv * 1.1, vec2(93.9898, 67.345))) * 23451.631);
    float grain = (n + n2) * 0.5 - 0.5;
    // Apply more grain to midtones, less to shadows/highlights
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float mask = 1.0 - abs(luma - 0.5) * 2.0;
    c.rgb += grain * intensity * mask * 0.1;
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

// ── Line Art Enhancement ────────────────────────────────────────

static const char* const kGlslThinLines = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Thin Lines (darken edges)

vec4 hook() {
    float strength = {{STRENGTH}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    float n = dot(HOOKED_texOff(vec2(0,-1)).rgb, vec3(0.2126,0.7152,0.0722));
    float s = dot(HOOKED_texOff(vec2(0, 1)).rgb, vec3(0.2126,0.7152,0.0722));
    float e = dot(HOOKED_texOff(vec2(1, 0)).rgb, vec3(0.2126,0.7152,0.0722));
    float w = dot(HOOKED_texOff(vec2(-1,0)).rgb, vec3(0.2126,0.7152,0.0722));
    float avg = (n + s + e + w) * 0.25;
    // Darken if center is darker than neighbours (line pixel)
    float diff = avg - luma;
    float darken = max(diff, 0.0) * strength;
    c.rgb -= darken;
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslBoldLines = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Bold Lines (thicken dark edges)

vec4 hook() {
    float strength = {{STRENGTH}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
    // Sample 3x3 for minimum (darkest neighbour)
    float mn = luma;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float l = dot(HOOKED_texOff(vec2(float(x),float(y))).rgb,
                         vec3(0.2126,0.7152,0.0722));
            mn = min(mn, l);
        }
    }
    float darkening = (luma - mn) * strength;
    c.rgb = mix(c.rgb, c.rgb - darkening, step(0.01, darkening));
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

static const char* const kGlslEdgeRefine = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Edge Refine (anti-alias lines)

vec4 hook() {
    float strength = {{STRENGTH}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    float luma = dot(c.rgb, vec3(0.2126,0.7152,0.0722));
    // Directional edge detection
    float hn = dot(HOOKED_texOff(vec2(-1,-1)).rgb, vec3(0.2126,0.7152,0.0722));
    float hc = dot(HOOKED_texOff(vec2( 0,-1)).rgb, vec3(0.2126,0.7152,0.0722));
    float hp = dot(HOOKED_texOff(vec2( 1,-1)).rgb, vec3(0.2126,0.7152,0.0722));
    float vn = dot(HOOKED_texOff(vec2(-1, 0)).rgb, vec3(0.2126,0.7152,0.0722));
    float vp = dot(HOOKED_texOff(vec2( 1, 0)).rgb, vec3(0.2126,0.7152,0.0722));
    float bn = dot(HOOKED_texOff(vec2(-1, 1)).rgb, vec3(0.2126,0.7152,0.0722));
    float bc = dot(HOOKED_texOff(vec2( 0, 1)).rgb, vec3(0.2126,0.7152,0.0722));
    float bp = dot(HOOKED_texOff(vec2( 1, 1)).rgb, vec3(0.2126,0.7152,0.0722));
    // Gradient magnitude
    float gx = (hp + 2.0*vp + bp) - (hn + 2.0*vn + bn);
    float gy = (bn + 2.0*bc + bp) - (hn + 2.0*hc + hp);
    float edge = sqrt(gx*gx + gy*gy);
    // Smooth edges
    vec3 avg = (HOOKED_texOff(vec2(-1,0)).rgb + HOOKED_texOff(vec2(1,0)).rgb +
                HOOKED_texOff(vec2(0,-1)).rgb + HOOKED_texOff(vec2(0,1)).rgb) * 0.25;
    c.rgb = mix(c.rgb, avg, edge * strength * 0.5);
    return vec4(clamp(c.rgb, 0.0, 1.0), c.a);
}
)";

// ── Deblurring ──────────────────────────────────────────────────

static const char* const kGlslDeblur = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Deblur (inverse sharpening)

vec4 hook() {
    float strength = {{STRENGTH}};
    int iterations = int({{ITERATIONS}});
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    vec3 result = c;
    for (int i = 0; i < iterations; i++) {
        vec3 blur = vec3(0.0);
        float total = 0.0;
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                float w = (x == 0 && y == 0) ? 4.0 : 1.0;
                blur += HOOKED_texOff(vec2(float(x),float(y))).rgb * w;
                total += w;
            }
        }
        blur /= total;
        result = result + (result - blur) * strength;
    }
    return vec4(clamp(result, 0.0, 1.0), 1.0);
}
)";

// ── Upscaling helpers ───────────────────────────────────────────

static const char* const kGlslKrigBilateral = R"(
//!HOOK CHROMA
//!BIND HOOKED
//!BIND LUMA
//!DESC Krig Bilateral Chroma Upscale

vec4 hook() {
    float sigma = {{SIGMA}};
    vec2 pt = HOOKED_pt;
    vec4 c = HOOKED_texOff(vec2(0,0));
    float lumaC = LUMA_texOff(vec2(0,0)).r;
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            float l = LUMA_texOff(vec2(float(x),float(y))).r;
            vec3 s = HOOKED_texOff(vec2(float(x),float(y))).rgb;
            float ld = l - lumaC;
            float spatial = float(x*x + y*y);
            float w = exp(-spatial / (2.0*4.0)) * exp(-ld*ld / (2.0*sigma*sigma));
            sum += s * w;
            wsum += w;
        }
    }
    return vec4(sum / max(wsum, 0.001), c.a);
}
)";

static const char* const kGlslSSimDownscaler = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC SSIM-aware Downscale Postprocess

vec4 hook() {
    float strength = {{STRENGTH}};
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    vec3 avg = vec3(0.0);
    vec3 avg2 = vec3(0.0);
    float count = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 s = HOOKED_texOff(vec2(float(x),float(y))).rgb;
            avg += s;
            avg2 += s * s;
            count += 1.0;
        }
    }
    avg /= count;
    avg2 /= count;
    vec3 variance = max(avg2 - avg*avg, 0.0);
    vec3 sigma = sqrt(variance);
    // Enhance detail that's above the noise floor
    vec3 detail = c - avg;
    vec3 enhanced = avg + detail * (1.0 + strength * sigma * 10.0);
    return vec4(clamp(enhanced, 0.0, 1.0), 1.0);
}
)";

// ── Restoration ─────────────────────────────────────────────────

static const char* const kGlslDeblockDCT = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Deblock (compression artifact removal)

vec4 hook() {
    float strength = {{STRENGTH}};
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    // Detect block boundaries (8x8 and 16x16 grid)
    vec2 pos = gl_FragCoord.xy;
    float block8  = step(0.5, max(
        1.0 - abs(mod(pos.x, 8.0) - 0.0) * 2.0,
        1.0 - abs(mod(pos.y, 8.0) - 0.0) * 2.0));
    // Apply stronger smoothing near block boundaries
    vec3 smooth4 = vec3(0.0);
    float w = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 s = HOOKED_texOff(vec2(float(x),float(y))).rgb;
            float wt = (x == 0 && y == 0) ? 2.0 : 1.0;
            smooth4 += s * wt;
            w += wt;
        }
    }
    smooth4 /= w;
    c = mix(c, smooth4, block8 * strength);
    return vec4(clamp(c, 0.0, 1.0), 1.0);
}
)";

static const char* const kGlslDeringMosquito = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Dering / Mosquito Noise Removal

vec4 hook() {
    float threshold = {{THRESHOLD}};
    float blend = {{BLEND}};
    vec3 c = HOOKED_texOff(vec2(0,0)).rgb;
    // Detect ringing: high local variance near strong edges
    float luma = dot(c, vec3(0.2126,0.7152,0.0722));
    vec3 mn = c, mx = c;
    float lavg = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 s = HOOKED_texOff(vec2(float(x),float(y))).rgb;
            mn = min(mn, s);
            mx = max(mx, s);
            lavg += dot(s, vec3(0.2126,0.7152,0.0722));
        }
    }
    lavg /= 9.0;
    float variance = dot(mx - mn, vec3(1.0)) / 3.0;
    float edge = abs(luma - lavg);
    float ring = smoothstep(threshold * 0.5, threshold, variance) *
                 smoothstep(0.0, threshold, edge);
    // Clamp to local range
    vec3 clamped = clamp(c, mn, mx);
    c = mix(c, clamped, ring * blend);
    return vec4(c, 1.0);
}
)";

// ── Utility ─────────────────────────────────────────────────────

static const char* const kGlslVignette = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Vignette

vec4 hook() {
    float strength = {{STRENGTH}};
    float radius = {{RADIUS}};
    vec4 c = HOOKED_texOff(vec2(0,0));
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(HOOKED_raw, 0));
    vec2 center = vec2(0.5);
    float dist = length(uv - center) / radius;
    float vignette = 1.0 - dist * dist * strength;
    c.rgb *= max(vignette, 0.0);
    return c;
}
)";

static const char* const kGlslChromaShift = R"(
//!HOOK MAIN
//!BIND HOOKED
//!DESC Chromatic Aberration Fix

vec4 hook() {
    float shift_r = {{SHIFT_R}};
    float shift_b = {{SHIFT_B}};
    vec2 center = vec2(0.5);
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(HOOKED_raw, 0));
    vec2 dir = uv - center;
    float r = HOOKED_tex(HOOKED_pos + dir * shift_r * 0.01).r;
    float g = HOOKED_texOff(vec2(0,0)).g;
    float b = HOOKED_tex(HOOKED_pos - dir * shift_b * 0.01).b;
    return vec4(r, g, b, 1.0);
}
)";

// ═════════════════════════════════════════════════════════════════
//  VapourSynth Script Sources
// ═════════════════════════════════════════════════════════════════

static const char* const kVsKnlMeansCl = R"(
# KNLMeansCL GPU Denoise
import vapoursynth as vs
core = vs.core
clip = video  # placeholder
d = int({{TEMPORAL_RADIUS}})
a = int({{SPATIAL_RADIUS}})
s = int({{SIMILARITY_RADIUS}})
h = {{H_STRENGTH}}
try:
    clip = core.knlm.KNLMeansCL(clip, d=d, a=a, s=s, h=h, device_type='gpu')
except:
    clip = core.knlm.KNLMeansCL(clip, d=d, a=a, s=s, h=h, device_type='cpu')
clip.set_output()
)";

static const char* const kVsBm3d = R"(
# BM3D Denoise
import vapoursynth as vs
core = vs.core
clip = video
sigma = {{SIGMA}}
radius = int({{TEMPORAL_RADIUS}})
try:
    clip_y = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    ref = core.bm3dcuda.BM3D(clip_y, sigma=[sigma], radius=radius)
    clip = core.bm3d.VAggregate(ref, radius=radius)
    clip = core.std.ShufflePlanes([clip, video], [0,1,2], vs.YUV)
except:
    try:
        clip = core.bm3d.Basic(clip, sigma=[sigma, sigma, sigma])
        clip = core.bm3d.Final(video, clip, sigma=[sigma, sigma, sigma])
    except:
        clip = core.bm3d.Basic(clip, sigma=[sigma, sigma, sigma])
clip.set_output()
)";

static const char* const kVsBm3dCuda = R"(
# BM3D CUDA/GPU Denoise (high quality)
import vapoursynth as vs
core = vs.core
clip = video
sigma = {{SIGMA}}
radius = int({{TEMPORAL_RADIUS}})
block_step = int({{BLOCK_STEP}})
bm_range = int({{BM_RANGE}})
try:
    clip = core.bm3dcuda.BM3D(clip, sigma=[sigma,sigma,sigma],
        radius=radius, block_step=block_step, bm_range=bm_range)
    clip = core.bm3d.VAggregate(clip, radius=radius)
except:
    clip = core.bm3d.Basic(clip, sigma=[sigma,sigma,sigma])
clip.set_output()
)";

static const char* const kVsDfttest = R"(
# DFTTest (frequency-domain denoising)
import vapoursynth as vs
core = vs.core
clip = video
sigma = {{SIGMA}}
tbsize = int({{TEMPORAL_SIZE}})
try:
    clip = core.dfttest.DFTTest(clip, sigma=sigma, tbsize=tbsize)
except:
    clip = core.dfttest.DFTTest(clip, sigma=sigma, tbsize=1)
clip.set_output()
)";

static const char* const kVsMvtoolsDenoise = R"(
# MVTools Temporal Denoise (motion-compensated)
import vapoursynth as vs
core = vs.core
clip = video
thsad = int({{THSAD}})
blksize = int({{BLOCK_SIZE}})
sup = core.mv.Super(clip, pel=2)
bv1 = core.mv.Analyse(sup, isb=True, delta=1, blksize=blksize)
fv1 = core.mv.Analyse(sup, isb=False, delta=1, blksize=blksize)
bv2 = core.mv.Analyse(sup, isb=True, delta=2, blksize=blksize)
fv2 = core.mv.Analyse(sup, isb=False, delta=2, blksize=blksize)
clip = core.mv.Degrain2(clip, sup, bv1, fv1, bv2, fv2, thsad=thsad)
clip.set_output()
)";

static const char* const kVsMvtoolsInterp = R"(
# MVTools Frame Interpolation (motion-compensated)
import vapoursynth as vs
core = vs.core
clip = video
target_fps_num = int({{TARGET_FPS}})
target_fps_den = 1
blksize = int({{BLOCK_SIZE}})
search = int({{SEARCH}})
sup = core.mv.Super(clip, pel=2)
bv = core.mv.Analyse(sup, isb=True, delta=1, blksize=blksize, search=search)
fv = core.mv.Analyse(sup, isb=False, delta=1, blksize=blksize, search=search)
clip = core.mv.FlowFPS(clip, sup, bv, fv, num=target_fps_num, den=target_fps_den)
clip.set_output()
)";

static const char* const kVsRife = R"(
# RIFE AI Frame Interpolation
import vapoursynth as vs
core = vs.core
clip = video
multi = int({{MULTIPLIER}})
try:
    from vsmlrt import RIFE, RIFEModel
    clip = RIFE(clip, multi=multi, model=RIFEModel.v4_6)
except:
    try:
        clip = core.rife.RIFE(clip, multiplier=multi)
    except:
        # Fallback to MVTools
        sup = core.mv.Super(clip, pel=2)
        bv = core.mv.Analyse(sup, isb=True)
        fv = core.mv.Analyse(sup, isb=False)
        clip = core.mv.FlowFPS(clip, sup, bv, fv,
            num=clip.fps.numerator * multi, den=clip.fps.denominator)
clip.set_output()
)";

static const char* const kVsNnedi3 = R"(
# NNEDI3 2x Upscale (neural network edge-directed interpolation)
import vapoursynth as vs
core = vs.core
clip = video
nsize = int({{NSIZE}})
nns = int({{NNS}})
qual = int({{QUALITY}})
try:
    clip = core.nnedi3cl.NNEDI3CL(clip, field=1, nsize=nsize, nns=nns, qual=qual)
    clip = core.std.Transpose(clip)
    clip = core.nnedi3cl.NNEDI3CL(clip, field=1, nsize=nsize, nns=nns, qual=qual)
    clip = core.std.Transpose(clip)
except:
    clip = core.nnedi3.nnedi3(clip, field=1, nsize=nsize, nns=nns, qual=qual)
    clip = core.std.Transpose(clip)
    clip = core.nnedi3.nnedi3(clip, field=1, nsize=nsize, nns=nns, qual=qual)
    clip = core.std.Transpose(clip)
clip.set_output()
)";

static const char* const kVsRealEsrgan = R"(
# Real-ESRGAN AI Upscale
import vapoursynth as vs
core = vs.core
clip = video
scale = int({{SCALE}})
try:
    from vsmlrt import NCNN, NCNNModel
    if scale == 4:
        clip = NCNN(clip, model=NCNNModel.RealESRGAN_x4)
    elif scale == 2:
        clip = NCNN(clip, model=NCNNModel.RealESRGAN_x2)
    else:
        clip = NCNN(clip, model=NCNNModel.RealESRGAN_x4)
        if scale != 4:
            w = clip.width * scale // 4
            h = clip.height * scale // 4
            clip = core.resize.Lanczos(clip, width=w, height=h)
except:
    w = clip.width * scale
    h = clip.height * scale
    clip = core.resize.Lanczos(clip, width=w, height=h)
clip.set_output()
)";

static const char* const kVsWaifu2x = R"(
# Waifu2x Upscale / Denoise
import vapoursynth as vs
core = vs.core
clip = video
noise_level = int({{NOISE_LEVEL}})
scale = int({{SCALE}})
try:
    from vsmlrt import NCNN, NCNNModel
    clip = NCNN(clip, model=NCNNModel.Waifu2x_CUnet_noise3_scale2x)
except:
    try:
        clip = core.w2xncnnvk.Waifu2x(clip, noise=noise_level, scale=scale)
    except:
        w = clip.width * scale
        h = clip.height * scale
        clip = core.resize.Lanczos(clip, width=w, height=h)
clip.set_output()
)";

static const char* const kVsDeblock = R"(
# Deblock (compression artifact removal)
import vapoursynth as vs
core = vs.core
clip = video
quant = int({{QUANT}})
aoffset = int({{A_OFFSET}})
boffset = int({{B_OFFSET}})
try:
    clip = core.deblock.Deblock(clip, quant=quant, aoffset=aoffset, boffset=boffset)
except:
    clip = core.std.Convolution(clip, [1,2,1,2,4,2,1,2,1])
clip.set_output()
)";

static const char* const kVsDehalo = R"(
# Fine Dehalo
import vapoursynth as vs
core = vs.core
clip = video
rx = {{RX}}
ry = {{RY}}
darkstr = {{DARK_STRENGTH}}
brightstr = {{BRIGHT_STRENGTH}}
try:
    import havsfunc as haf
    clip = haf.FineDehalo(clip, rx=rx, ry=ry, darkstr=darkstr, brightstr=brightstr)
except:
    # Fallback: simple max/min clamp
    mx = core.std.Maximum(clip)
    mn = core.std.Minimum(clip)
    clip = core.std.Expr([clip, mn, mx], 'x y max z min')
clip.set_output()
)";

static const char* const kVsHistogramEq = R"(
# Histogram Equalization (auto levels)
import vapoursynth as vs
core = vs.core
clip = video
try:
    clip = core.hist.Equalize(clip)
except:
    clip = core.std.Levels(clip, min_in=16, max_in=235, min_out=0, max_out=255)
clip.set_output()
)";

static const char* const kVsAutoWhiteBalance = R"(
# Auto White Balance (gray world assumption)
import vapoursynth as vs
core = vs.core
clip = video
strength = {{STRENGTH}}
try:
    import adjust
    clip = adjust.Tweak(clip, sat=1.0 + strength * 0.2, cont=1.0 + strength * 0.1)
except:
    clip = core.std.Levels(clip, min_in=16, max_in=235, min_out=0, max_out=255)
clip.set_output()
)";

static const char* const kVsCTMF = R"(
# CTMF Median Filter (fast constant-time)
import vapoursynth as vs
core = vs.core
clip = video
radius = int({{RADIUS}})
try:
    clip = core.ctmf.CTMF(clip, radius=radius)
except:
    clip = core.std.Median(clip)
clip.set_output()
)";

static const char* const kVsRemoveGrain = R"(
# RemoveGrain (multiple modes)
import vapoursynth as vs
core = vs.core
clip = video
mode = int({{MODE}})
clip = core.rgvs.RemoveGrain(clip, mode=[mode])
clip.set_output()
)";

static const char* const kVsTemporalSoften = R"(
# Temporal Soften (temporal averaging)
import vapoursynth as vs
core = vs.core
clip = video
radius = int({{RADIUS}})
luma_threshold = int({{LUMA_THRESHOLD}})
chroma_threshold = int({{CHROMA_THRESHOLD}})
try:
    clip = core.focus2.TemporalSoften2(clip, radius=radius,
        luma_threshold=luma_threshold, chroma_threshold=chroma_threshold)
except:
    clip = core.focus.TemporalSoften(clip, radius=radius,
        luma_threshold=luma_threshold, chroma_threshold=chroma_threshold)
clip.set_output()
)";

// ═════════════════════════════════════════════════════════════════
//  Filter catalog construction
// ═════════════════════════════════════════════════════════════════

namespace {

std::vector<EmbeddedFilter> buildCatalog() {
    std::vector<EmbeddedFilter> filters;

    // ── GLSL Sharpening ─────────────────────────────────────────

    filters.push_back({
        "glsl.cas",
        "Contrast Adaptive Sharpening",
        "AMD FidelityFX CAS-style adaptive sharpening that avoids over-sharpening edges.",
        FilterCategory::Sharpen, FilterRuntime::Glsl, StageKind::Sharpen, 10,
        kGlslCAS,
        {{"SHARPNESS", "Sharpness", "Overall sharpening intensity", 0.0, 1.0, 0.4, 0.05, false}}
    });

    filters.push_back({
        "glsl.adaptive_sharpen",
        "Adaptive Sharpen",
        "Edge-aware sharpening with Sobel edge detection and variable strength.",
        FilterCategory::Sharpen, FilterRuntime::Glsl, StageKind::Sharpen, 20,
        kGlslAdaptiveSharpen,
        {{"CURVE_HEIGHT", "Curve Height", "Sharpening curve intensity", 0.1, 3.0, 0.7, 0.1, false}}
    });

    filters.push_back({
        "glsl.unsharp_mask",
        "Unsharp Mask",
        "Classic unsharp masking with adjustable radius and amount.",
        FilterCategory::Sharpen, FilterRuntime::Glsl, StageKind::Sharpen, 30,
        kGlslUnsharpMask,
        {{"AMOUNT", "Amount", "Sharpening multiplier", 0.1, 5.0, 1.5, 0.1, false},
         {"RADIUS", "Radius", "Blur radius for the mask", 0.5, 5.0, 1.5, 0.1, false}}
    });

    filters.push_back({
        "glsl.luma_sharpen",
        "Luma Sharpen",
        "Luminance-only sharpening to avoid colour fringing.",
        FilterCategory::Sharpen, FilterRuntime::Glsl, StageKind::Sharpen, 40,
        kGlslLumaSharpen,
        {{"STRENGTH", "Strength", "Sharpening strength", 0.1, 3.0, 1.0, 0.1, false},
         {"CLAMP", "Clamp", "Maximum allowed change per pixel", 0.01, 1.0, 0.2, 0.01, false}}
    });

    // ── GLSL Denoising ──────────────────────────────────────────

    filters.push_back({
        "glsl.nlmeans",
        "Non-Local Means Denoise",
        "Advanced denoising that compares patches across the image. Excellent quality, moderate speed.",
        FilterCategory::Denoise, FilterRuntime::Glsl, StageKind::Denoise, 10,
        kGlslNlMeans,
        {{"H_STRENGTH", "Strength (h)", "Filter strength — higher removes more noise but may blur", 0.01, 0.5, 0.1, 0.01, false}}
    });

    filters.push_back({
        "glsl.bilateral",
        "Bilateral Denoise",
        "Edge-preserving smoothing that respects colour boundaries.",
        FilterCategory::Denoise, FilterRuntime::Glsl, StageKind::Denoise, 20,
        kGlslBilateralDenoise,
        {{"SPATIAL_SIGMA", "Spatial Sigma", "Spatial blur radius", 0.5, 5.0, 1.5, 0.1, false},
         {"RANGE_SIGMA", "Range Sigma", "Intensity difference tolerance", 0.01, 0.5, 0.1, 0.01, false}}
    });

    filters.push_back({
        "glsl.median3x3",
        "Median 3×3 Denoise",
        "Fast median filter — excellent for salt-and-pepper noise without blurring edges.",
        FilterCategory::Denoise, FilterRuntime::Glsl, StageKind::Denoise, 30,
        kGlslMedianDenoise,
        {{"BLEND", "Blend", "Mix between original and median-filtered output", 0.0, 1.0, 0.8, 0.05, false}}
    });

    filters.push_back({
        "glsl.temporal_denoise",
        "Temporal Averaging",
        "Lightweight spatial approximation of temporal denoising.",
        FilterCategory::Denoise, FilterRuntime::Glsl, StageKind::Denoise, 40,
        kGlslTemporalDenoise,
        {{"STRENGTH", "Strength", "Averaging threshold adaptation", 0.01, 1.0, 0.3, 0.01, false}}
    });

    // ── GLSL Deband ─────────────────────────────────────────────

    filters.push_back({
        "glsl.deband",
        "Deband",
        "Removes banding artifacts from gradients (common in compressed video).",
        FilterCategory::Restoration, FilterRuntime::Glsl, StageKind::RemoveArtifacts, 10,
        kGlslDeband,
        {{"THRESHOLD", "Threshold", "Banding detection threshold (0-255 scale)", 16.0, 96.0, 48.0, 1.0, true},
         {"RANGE", "Range", "Sampling radius for comparing pixels", 1.0, 8.0, 4.0, 1.0, true},
         {"GRAIN", "Grain", "Dither strength to mask remaining bands", 0.0, 48.0, 6.0, 1.0, true}}
    });

    // ── GLSL Dehaloing ──────────────────────────────────────────

    filters.push_back({
        "glsl.dehalo",
        "Dehalo",
        "Removes halo artifacts from over-sharpened or upscaled content.",
        FilterCategory::Dehalo, FilterRuntime::Glsl, StageKind::Dehalo, 10,
        kGlslDehalo,
        {{"STRENGTH", "Strength", "Dehalo intensity", 0.1, 2.0, 1.0, 0.1, false}}
    });

    filters.push_back({
        "glsl.fine_edge_dehalo",
        "Fine Edge Dehalo",
        "Precise halo removal with separate bright/dark overshoot control.",
        FilterCategory::Dehalo, FilterRuntime::Glsl, StageKind::Dehalo, 20,
        kGlslFineEdgeDehalo,
        {{"RADIUS", "Radius", "Search radius for min/max", 1.0, 4.0, 2.0, 1.0, true},
         {"DARK_STRENGTH", "Dark Strength", "Undershoot correction", 0.0, 2.0, 0.5, 0.1, false},
         {"BRIGHT_STRENGTH", "Bright Strength", "Overshoot correction", 0.0, 2.0, 0.5, 0.1, false}}
    });

    // ── GLSL Color Correction ────────────────────────────────────

    filters.push_back({
        "glsl.color_balance",
        "Color Balance (Lift/Gain)",
        "Per-channel lift and gain controls for colour grading.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 10,
        kGlslColorBalance,
        {{"LIFT_R", "Lift Red", "Red offset", -0.5, 0.5, 0.0, 0.01, false},
         {"LIFT_G", "Lift Green", "Green offset", -0.5, 0.5, 0.0, 0.01, false},
         {"LIFT_B", "Lift Blue", "Blue offset", -0.5, 0.5, 0.0, 0.01, false},
         {"GAIN_R", "Gain Red", "Red multiplier", 0.5, 2.0, 1.0, 0.01, false},
         {"GAIN_G", "Gain Green", "Green multiplier", 0.5, 2.0, 1.0, 0.01, false},
         {"GAIN_B", "Gain Blue", "Blue multiplier", 0.5, 2.0, 1.0, 0.01, false}}
    });

    filters.push_back({
        "glsl.vibrance",
        "Vibrance",
        "Intelligent saturation that boosts muted colours more than already-saturated ones.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 20,
        kGlslVibrance,
        {{"VIBRANCE", "Vibrance", "Boost amount", -1.0, 2.0, 0.5, 0.05, false}}
    });

    filters.push_back({
        "glsl.auto_levels",
        "Auto Levels",
        "Stretch histogram by setting black and white points.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 30,
        kGlslAutoLevels,
        {{"BLACK_POINT", "Black Point", "Input black level (0-255)", 0.0, 64.0, 16.0, 1.0, true},
         {"WHITE_POINT", "White Point", "Input white level (0-255)", 200.0, 255.0, 235.0, 1.0, true}}
    });

    filters.push_back({
        "glsl.gamma",
        "Gamma Correction",
        "Adjust overall brightness curve.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 40,
        kGlslGammaCorrection,
        {{"GAMMA", "Gamma", "Gamma value (< 1 = darker, > 1 = brighter)", 0.2, 3.0, 1.0, 0.05, false}}
    });

    filters.push_back({
        "glsl.saturation",
        "Saturation Boost",
        "Simple saturation multiplier.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 50,
        kGlslSaturationBoost,
        {{"SATURATION", "Saturation", "Saturation multiplier (1.0 = no change)", 0.0, 3.0, 1.2, 0.05, false}}
    });

    filters.push_back({
        "glsl.contrast_curve",
        "S-Curve Contrast",
        "Sigmoid-based contrast enhancement.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 60,
        kGlslContrastCurve,
        {{"CONTRAST", "Contrast", "Curve steepness", 0.5, 3.0, 1.0, 0.05, false},
         {"MIDPOINT", "Midpoint", "Center of the curve", 0.3, 0.7, 0.5, 0.01, false}}
    });

    filters.push_back({
        "glsl.color_temperature",
        "Color Temperature",
        "Shift white balance between warm and cool tones.",
        FilterCategory::ColorCorrection, FilterRuntime::Glsl, StageKind::ColorFix, 70,
        kGlslColorTemperature,
        {{"TEMPERATURE", "Temperature", "Warm (+) / Cool (-) shift", -3.0, 3.0, 0.0, 0.1, false},
         {"TINT", "Tint", "Green (+) / Magenta (-) shift", -3.0, 3.0, 0.0, 0.1, false}}
    });

    // ── GLSL Film Grain ─────────────────────────────────────────

    filters.push_back({
        "glsl.film_grain",
        "Film Grain",
        "Add organic film-like grain texture, with midtone-weighted application.",
        FilterCategory::Grain, FilterRuntime::Glsl, StageKind::Sharpen, 90,
        kGlslFilmGrain,
        {{"INTENSITY", "Intensity", "Grain visibility", 0.0, 3.0, 0.5, 0.05, false},
         {"GRAIN_SIZE", "Size", "Grain particle size (pixels)", 1.0, 8.0, 2.0, 0.5, false}}
    });

    // ── GLSL Line Art ───────────────────────────────────────────

    filters.push_back({
        "glsl.thin_lines",
        "Thin Lines",
        "Darken detected line edges (great for animation / anime).",
        FilterCategory::LineArt, FilterRuntime::Glsl, StageKind::Sharpen, 50,
        kGlslThinLines,
        {{"STRENGTH", "Strength", "Line darkening intensity", 0.1, 3.0, 1.2, 0.1, false}}
    });

    filters.push_back({
        "glsl.bold_lines",
        "Bold Lines",
        "Thicken dark edges by applying minimum-based expansion.",
        FilterCategory::LineArt, FilterRuntime::Glsl, StageKind::Sharpen, 60,
        kGlslBoldLines,
        {{"STRENGTH", "Strength", "Line bolding intensity", 0.1, 2.0, 0.8, 0.1, false}}
    });

    filters.push_back({
        "glsl.edge_refine",
        "Edge Refine",
        "Anti-alias and smooth jagged line edges using directional detection.",
        FilterCategory::LineArt, FilterRuntime::Glsl, StageKind::Sharpen, 70,
        kGlslEdgeRefine,
        {{"STRENGTH", "Strength", "Edge smoothing amount", 0.1, 2.0, 0.6, 0.1, false}}
    });

    // ── GLSL Deblurring ─────────────────────────────────────────

    filters.push_back({
        "glsl.deblur",
        "Deblur",
        "Iterative inverse-blur sharpening for slightly out-of-focus content.",
        FilterCategory::Deblur, FilterRuntime::Glsl, StageKind::Deblur, 10,
        kGlslDeblur,
        {{"STRENGTH", "Strength", "Deblur correction amount", 0.1, 1.0, 0.3, 0.05, false},
         {"ITERATIONS", "Iterations", "Number of correction passes", 1.0, 5.0, 2.0, 1.0, true}}
    });

    // ── GLSL Upscale helpers ────────────────────────────────────

    filters.push_back({
        "glsl.krig_bilateral",
        "KrigBilateral Chroma Upscale",
        "Luma-guided bilateral upscaling for chroma planes.",
        FilterCategory::Upscale, FilterRuntime::Glsl, StageKind::Upscale, 80,
        kGlslKrigBilateral,
        {{"SIGMA", "Sigma", "Luma difference tolerance", 0.01, 1.0, 0.3, 0.01, false}}
    });

    filters.push_back({
        "glsl.ssim_downscaler",
        "SSIM-aware Detail Enhance",
        "Structural similarity based detail preservation after scaling.",
        FilterCategory::Upscale, FilterRuntime::Glsl, StageKind::Upscale, 90,
        kGlslSSimDownscaler,
        {{"STRENGTH", "Strength", "Detail enhancement factor", 0.0, 2.0, 0.5, 0.05, false}}
    });

    // ── GLSL Restoration ────────────────────────────────────────

    filters.push_back({
        "glsl.deblock",
        "Deblock (GLSL)",
        "Remove 8×8 block boundary artifacts from MPEG/H.264 compression.",
        FilterCategory::Restoration, FilterRuntime::Glsl, StageKind::RestoreCompression, 10,
        kGlslDeblockDCT,
        {{"STRENGTH", "Strength", "Smoothing intensity at block edges", 0.1, 2.0, 0.8, 0.1, false}}
    });

    filters.push_back({
        "glsl.dering",
        "Dering / Mosquito Noise",
        "Remove ringing artifacts near hard edges in compressed video.",
        FilterCategory::Restoration, FilterRuntime::Glsl, StageKind::RemoveArtifacts, 20,
        kGlslDeringMosquito,
        {{"THRESHOLD", "Threshold", "Ringing detection sensitivity", 0.01, 0.5, 0.1, 0.01, false},
         {"BLEND", "Blend", "Correction strength", 0.1, 1.0, 0.6, 0.05, false}}
    });

    // ── GLSL Utility ────────────────────────────────────────────

    filters.push_back({
        "glsl.vignette",
        "Vignette",
        "Darken corners for a cinematic look.",
        FilterCategory::Utility, FilterRuntime::Glsl, StageKind::ColorFix, 90,
        kGlslVignette,
        {{"STRENGTH", "Strength", "Vignette darkness", 0.0, 3.0, 0.5, 0.1, false},
         {"RADIUS", "Radius", "Vignette radius (larger = tighter)", 0.3, 1.5, 0.75, 0.05, false}}
    });

    filters.push_back({
        "glsl.chromatic_aberration_fix",
        "Chromatic Aberration Fix",
        "Correct red/blue fringing at image edges.",
        FilterCategory::Utility, FilterRuntime::Glsl, StageKind::ColorFix, 95,
        kGlslChromaShift,
        {{"SHIFT_R", "Red Shift", "Red channel radial offset", -5.0, 5.0, 0.0, 0.1, false},
         {"SHIFT_B", "Blue Shift", "Blue channel radial offset", -5.0, 5.0, 0.0, 0.1, false}}
    });

    // ═════════════════════════════════════════════════════════════
    //  VapourSynth Filters
    // ═════════════════════════════════════════════════════════════

    filters.push_back({
        "vs.knlmeanscl",
        "KNLMeansCL (GPU Denoise)",
        "OpenCL non-local means denoising with GPU acceleration.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 10,
        kVsKnlMeansCl,
        {{"TEMPORAL_RADIUS", "Temporal Radius", "Frames to average (0 = spatial only)", 0.0, 4.0, 1.0, 1.0, true},
         {"SPATIAL_RADIUS", "Spatial Radius", "Pixel search area", 1.0, 4.0, 2.0, 1.0, true},
         {"SIMILARITY_RADIUS", "Similarity Radius", "Patch comparison size", 1.0, 8.0, 4.0, 1.0, true},
         {"H_STRENGTH", "Strength (h)", "Filter strength", 0.1, 10.0, 1.2, 0.1, false}}
    });

    filters.push_back({
        "vs.bm3d",
        "BM3D Denoise",
        "Block-matching 3D denoising — gold standard for quality.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 20,
        kVsBm3d,
        {{"SIGMA", "Sigma", "Noise standard deviation estimate", 1.0, 30.0, 5.0, 0.5, false},
         {"TEMPORAL_RADIUS", "Temporal Radius", "Frames for temporal averaging", 0.0, 4.0, 1.0, 1.0, true}}
    });

    filters.push_back({
        "vs.bm3d_cuda",
        "BM3D CUDA/GPU Denoise",
        "GPU-accelerated BM3D with tunable block and search parameters.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 25,
        kVsBm3dCuda,
        {{"SIGMA", "Sigma", "Noise estimate", 1.0, 30.0, 5.0, 0.5, false},
         {"TEMPORAL_RADIUS", "Temporal Radius", "Temporal frames", 0.0, 4.0, 1.0, 1.0, true},
         {"BLOCK_STEP", "Block Step", "Block stride (lower = better but slower)", 1.0, 8.0, 4.0, 1.0, true},
         {"BM_RANGE", "BM Range", "Block-matching search range", 4.0, 32.0, 16.0, 2.0, true}}
    });

    filters.push_back({
        "vs.dfttest",
        "DFTTest (Frequency Denoise)",
        "Frequency-domain denoising via discrete Fourier transform.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 30,
        kVsDfttest,
        {{"SIGMA", "Sigma", "Frequency-domain threshold", 1.0, 50.0, 8.0, 1.0, false},
         {"TEMPORAL_SIZE", "Temporal Size", "Block size for temporal dimension (1 = spatial only)", 1.0, 5.0, 1.0, 2.0, true}}
    });

    filters.push_back({
        "vs.mvtools_denoise",
        "MVTools Temporal Denoise",
        "Motion-compensated temporal denoising using optical flow.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 40,
        kVsMvtoolsDenoise,
        {{"THSAD", "ThSAD", "SAD threshold for motion detection", 100.0, 1000.0, 300.0, 50.0, true},
         {"BLOCK_SIZE", "Block Size", "Motion search block size", 4.0, 32.0, 16.0, 4.0, true}}
    });

    filters.push_back({
        "vs.ctmf",
        "CTMF Median Filter",
        "Constant-time median filter — very fast for large radii.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 50,
        kVsCTMF,
        {{"RADIUS", "Radius", "Filter radius", 1.0, 8.0, 2.0, 1.0, true}}
    });

    filters.push_back({
        "vs.removegrain",
        "RemoveGrain",
        "Multi-mode spatial grain/noise remover (17 modes available).",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 60,
        kVsRemoveGrain,
        {{"MODE", "Mode", "RG mode (1-24, see docs)", 1.0, 24.0, 17.0, 1.0, true}}
    });

    filters.push_back({
        "vs.temporal_soften",
        "Temporal Soften",
        "Simple temporal averaging between consecutive frames.",
        FilterCategory::Denoise, FilterRuntime::VapourSynth, StageKind::Denoise, 70,
        kVsTemporalSoften,
        {{"RADIUS", "Radius", "Number of surrounding frames", 1.0, 7.0, 2.0, 1.0, true},
         {"LUMA_THRESHOLD", "Luma Threshold", "Luma similarity threshold", 1.0, 255.0, 4.0, 1.0, true},
         {"CHROMA_THRESHOLD", "Chroma Threshold", "Chroma similarity threshold", 1.0, 255.0, 8.0, 1.0, true}}
    });

    // ── VS Frame Interpolation ──────────────────────────────────

    filters.push_back({
        "vs.mvtools_interp",
        "MVTools Interpolation",
        "Motion-compensated frame doubling/tripling using optical flow.",
        FilterCategory::Interpolation, FilterRuntime::VapourSynth, StageKind::Interpolate, 10,
        kVsMvtoolsInterp,
        {{"TARGET_FPS", "Target FPS", "Desired output framerate", 24.0, 120.0, 60.0, 1.0, true},
         {"BLOCK_SIZE", "Block Size", "Motion search block size", 4.0, 32.0, 16.0, 4.0, true},
         {"SEARCH", "Search Mode", "Search algorithm (0=onetime, 3=exhaustive)", 0.0, 5.0, 3.0, 1.0, true}}
    });

    filters.push_back({
        "vs.rife",
        "RIFE AI Interpolation",
        "Neural-network frame interpolation for ultra-smooth motion.",
        FilterCategory::Interpolation, FilterRuntime::VapourSynth, StageKind::Interpolate, 20,
        kVsRife,
        {{"MULTIPLIER", "Multiplier", "Frame multiplier (2 = double, 4 = quadruple)", 2.0, 8.0, 2.0, 1.0, true}}
    });

    // ── VS Upscale ──────────────────────────────────────────────

    filters.push_back({
        "vs.nnedi3",
        "NNEDI3 2× Upscale",
        "Neural network edge-directed interpolation for crisp 2× upscaling.",
        FilterCategory::Upscale, FilterRuntime::VapourSynth, StageKind::Upscale, 10,
        kVsNnedi3,
        {{"NSIZE", "Network Size", "Size of the local neighbourhood (0-6)", 0.0, 6.0, 4.0, 1.0, true},
         {"NNS", "Neurons", "Number of neurons (0-4, higher = better + slower)", 0.0, 4.0, 3.0, 1.0, true},
         {"QUALITY", "Quality", "Quality level (1 = fast, 2 = best)", 1.0, 2.0, 1.0, 1.0, true}}
    });

    filters.push_back({
        "vs.real_esrgan",
        "Real-ESRGAN Upscale",
        "AI super-resolution using Real-ESRGAN via vs-mlrt NCNN backend.",
        FilterCategory::Upscale, FilterRuntime::VapourSynth, StageKind::Upscale, 20,
        kVsRealEsrgan,
        {{"SCALE", "Scale Factor", "Upscale multiplier (2 or 4)", 2.0, 4.0, 4.0, 2.0, true}}
    });

    filters.push_back({
        "vs.waifu2x",
        "Waifu2x Upscale/Denoise",
        "Convolutional neural network upscaling optimised for drawn content.",
        FilterCategory::Upscale, FilterRuntime::VapourSynth, StageKind::Upscale, 30,
        kVsWaifu2x,
        {{"NOISE_LEVEL", "Noise Level", "Denoise intensity (0-3)", 0.0, 3.0, 1.0, 1.0, true},
         {"SCALE", "Scale Factor", "Upscale multiplier", 1.0, 4.0, 2.0, 1.0, true}}
    });

    // ── VS Restoration ──────────────────────────────────────────

    filters.push_back({
        "vs.deblock",
        "Deblock (VapourSynth)",
        "Remove MPEG/H.264 block boundary artifacts.",
        FilterCategory::Restoration, FilterRuntime::VapourSynth, StageKind::RestoreCompression, 10,
        kVsDeblock,
        {{"QUANT", "Quant", "Deblocking quantizer (higher = stronger)", 10.0, 60.0, 25.0, 1.0, true},
         {"A_OFFSET", "A Offset", "Alpha offset for boundary strength", 0.0, 20.0, 0.0, 1.0, true},
         {"B_OFFSET", "B Offset", "Beta offset for boundary detail", 0.0, 20.0, 0.0, 1.0, true}}
    });

    // ── VS Dehalo ───────────────────────────────────────────────

    filters.push_back({
        "vs.fine_dehalo",
        "Fine Dehalo (VapourSynth)",
        "Advanced halo removal with separate dark/bright controls.",
        FilterCategory::Dehalo, FilterRuntime::VapourSynth, StageKind::Dehalo, 10,
        kVsDehalo,
        {{"RX", "RX", "Horizontal dehalo radius", 1.0, 4.0, 2.0, 0.5, false},
         {"RY", "RY", "Vertical dehalo radius", 1.0, 4.0, 2.0, 0.5, false},
         {"DARK_STRENGTH", "Dark Strength", "Dark halo removal", 0.0, 1.0, 0.3, 0.1, false},
         {"BRIGHT_STRENGTH", "Bright Strength", "Bright halo removal", 0.0, 1.0, 0.5, 0.1, false}}
    });

    // ── VS Color Correction ─────────────────────────────────────

    filters.push_back({
        "vs.histogram_eq",
        "Auto Histogram Equalization",
        "Automatically stretch the histogram for maximum dynamic range.",
        FilterCategory::ColorCorrection, FilterRuntime::VapourSynth, StageKind::ColorFix, 10,
        kVsHistogramEq,
        {}
    });

    filters.push_back({
        "vs.auto_wb",
        "Auto White Balance",
        "Correct white-balance drift using gray-world assumption.",
        FilterCategory::ColorCorrection, FilterRuntime::VapourSynth, StageKind::ColorFix, 20,
        kVsAutoWhiteBalance,
        {{"STRENGTH", "Strength", "Correction intensity", 0.1, 2.0, 1.0, 0.1, false}}
    });

    return filters;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
//  Public API
// ═════════════════════════════════════════════════════════════════

const std::vector<EmbeddedFilter>& allEmbeddedFilters() {
    static const std::vector<EmbeddedFilter> catalog = buildCatalog();
    return catalog;
}

const EmbeddedFilter* findFilter(const std::string& id) {
    for (const auto& f : allEmbeddedFilters()) {
        if (f.id == id) { return &f; }
    }
    return nullptr;
}

std::vector<const EmbeddedFilter*> filtersForCategory(FilterCategory cat) {
    std::vector<const EmbeddedFilter*> result;
    for (const auto& f : allEmbeddedFilters()) {
        if (f.category == cat) { result.push_back(&f); }
    }
    std::sort(result.begin(), result.end(),
              [](const EmbeddedFilter* a, const EmbeddedFilter* b) {
                  return a->sortOrder < b->sortOrder;
              });
    return result;
}

std::vector<const EmbeddedFilter*> filtersForRuntime(FilterRuntime rt) {
    std::vector<const EmbeddedFilter*> result;
    for (const auto& f : allEmbeddedFilters()) {
        if (f.runtime == rt) { result.push_back(&f); }
    }
    std::sort(result.begin(), result.end(),
              [](const EmbeddedFilter* a, const EmbeddedFilter* b) {
                  return a->sortOrder < b->sortOrder;
              });
    return result;
}

std::vector<const EmbeddedFilter*> filtersForStage(StageKind kind) {
    std::vector<const EmbeddedFilter*> result;
    for (const auto& f : allEmbeddedFilters()) {
        if (f.stageKind == kind) { result.push_back(&f); }
    }
    std::sort(result.begin(), result.end(),
              [](const EmbeddedFilter* a, const EmbeddedFilter* b) {
                  return a->sortOrder < b->sortOrder;
              });
    return result;
}

std::string resolveSource(const EmbeddedFilter& filter,
                          const std::unordered_map<std::string, double>& paramValues) {
    std::string src = filter.source;
    // For each parameter, replace {{KEY}} with the value (or default).
    for (const auto& pd : filter.params) {
        double val = pd.defVal;
        auto it = paramValues.find(pd.key);
        if (it != paramValues.end()) { val = it->second; }
        // Clamp to range.
        val = std::max(pd.minVal, std::min(pd.maxVal, val));

        std::string placeholder = "{{" + pd.key + "}}";
        std::string replacement;
        if (pd.isInt) {
            replacement = std::to_string(static_cast<int>(val));
        } else {
            std::ostringstream oss;
            oss << val;
            replacement = oss.str();
        }
        // Replace all occurrences.
        std::size_t pos = 0;
        while ((pos = src.find(placeholder, pos)) != std::string::npos) {
            src.replace(pos, placeholder.size(), replacement);
            pos += replacement.size();
        }
    }
    return src;
}

}  // namespace ave

#endif // ── entire file disabled ──────────────────────────────
