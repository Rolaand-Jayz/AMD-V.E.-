cmake_minimum_required(VERSION 3.21)

project(amd_video_enhancer_cpp LANGUAGES CXX)

include(GNUInstallDirs)
include(cmake/ave_rocm_discovery.cmake)

set(AVE_ROCM_ROOT_HINTS "" CACHE STRING
    "Additional ROCm installation prefixes to search (semicolon-separated)")
ave_discover_rocm_prefixes()
if(AVE_ROCM_DISCOVERED_PREFIXES)
    message(STATUS "ROCm prefixes available to CMake: ${AVE_ROCM_DISCOVERED_PREFIXES}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─────────────────────────────────────────────────────────────────
# Build options
# ─────────────────────────────────────────────────────────────────
option(AVE_STRICT_WARNINGS   "Enable strict warning flags"          ON)
option(AVE_BUILD_GUI         "Build Qt desktop GUI"                  ON)
option(AVE_HAVE_CURL         "Enable libcurl for model downloads"    ON)
option(AVE_HAVE_MIGRAPHX     "Enable MiGraphX ROCm runtime"          OFF)
option(AVE_HAVE_HIP          "Enable HIP GPU headers (ROCm)"         OFF)
option(AVE_HAVE_ONNXRUNTIME_ROCM
    "Enable ONNX Runtime ROCm/HIP fallback runtime"               ON)
option(AVE_HAVE_NCNN         "Enable NCNN Vulkan runtime"            OFF)
option(AVE_HAVE_VULKAN       "Enable Vulkan runtime for interop"     ON)
# ROCTx: zero-overhead stage markers for rocprof --roctx-trace integration.
# Requires roctracer-dev package: sudo apt install roctracer-dev
option(AVE_HAVE_ROCTX        "Enable ROCTx markers for rocprof"      OFF)
option(AVE_INSTALL_BUNDLED_MIGRAPHX
    "Install the bundled MiGraphX compiler toolchain with the app" OFF)
option(AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER
    "Install bundled MiGraphX compiler tools and ONNX/TF sidecars with the app" OFF)
option(AVE_PREFER_BUNDLED_MIGRAPHX_FOR_BUILD
       "Prefer AVE_BUNDLED_MIGRAPHX_PREFIX for build-time MiGraphX discovery" OFF)
option(AVE_STRICT_BUNDLED_MIGRAPHX
       "Fail packaging when the bundled MiGraphX runtime is missing transitive shared libraries" ON)
option(AVE_INSTALL_BUNDLED_MODELS
       "Install the full model catalog with the app"                 ON)
option(AVE_INSTALL_RUNTIME_DEPS
       "Install non-system runtime shared libraries with the app"    ON)

set(_AVE_DEFAULT_BUNDLED_MIGRAPHX_PREFIX "$ENV{AVE_BUNDLED_MIGRAPHX_PREFIX}")
set(AVE_BUNDLED_MIGRAPHX_PREFIX "${_AVE_DEFAULT_BUNDLED_MIGRAPHX_PREFIX}" CACHE PATH
    "Explicit custom MiGraphX install prefix to bundle with the app")
set(_AVE_DEFAULT_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS "")
foreach(_ave_extra_root IN ITEMS /usr/lib /usr/local/lib /opt/rocm/lib /opt/rocm/lib64)
    if(EXISTS "${_ave_extra_root}")
        list(APPEND _AVE_DEFAULT_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS "${_ave_extra_root}")
    endif()
endforeach()
set(AVE_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS
    "${_AVE_DEFAULT_BUNDLED_MIGRAPHX_EXTRA_LIBRARY_DIRS}" CACHE STRING
    "Extra directories searched when bundling transitive MiGraphX compiler runtime libraries")

if(AVE_HAVE_MIGRAPHX AND AVE_PREFER_BUNDLED_MIGRAPHX_FOR_BUILD AND
   NOT AVE_BUNDLED_MIGRAPHX_PREFIX STREQUAL "")
    list(FIND CMAKE_PREFIX_PATH "${AVE_BUNDLED_MIGRAPHX_PREFIX}" _ave_migraphx_prefix_index)
    if(_ave_migraphx_prefix_index EQUAL -1)
        list(PREPEND CMAKE_PREFIX_PATH "${AVE_BUNDLED_MIGRAPHX_PREFIX}")
    endif()
    message(STATUS "Preferring optimized MiGraphX runtime from ${AVE_BUNDLED_MIGRAPHX_PREFIX}")
endif()

if(AVE_INSTALL_BUNDLED_MIGRAPHX)
    if(AVE_BUNDLED_MIGRAPHX_PREFIX STREQUAL "")
        message(STATUS
            "AVE_INSTALL_BUNDLED_MIGRAPHX=ON but no explicit AVE_BUNDLED_MIGRAPHX_PREFIX was provided; "
            "disabling bundled MiGraphX install for this configure.")
        set(AVE_INSTALL_BUNDLED_MIGRAPHX OFF)
        set(AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER OFF)
    elseif(NOT EXISTS "${AVE_BUNDLED_MIGRAPHX_PREFIX}/lib/migraphx/lib")
        message(WARNING
            "AVE_INSTALL_BUNDLED_MIGRAPHX=ON but AVE_BUNDLED_MIGRAPHX_PREFIX does not contain "
            "lib/migraphx/lib; disabling bundled MiGraphX install.")
        set(AVE_INSTALL_BUNDLED_MIGRAPHX OFF)
        set(AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER OFF)
    else()
        message(STATUS "Bundling optimized MiGraphX runtime from ${AVE_BUNDLED_MIGRAPHX_PREFIX}")
        if(AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER AND
           NOT EXISTS "${AVE_BUNDLED_MIGRAPHX_PREFIX}/bin/migraphx-driver")
            message(WARNING
                "AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER=ON but the bundled MiGraphX prefix does "
                "not contain bin/migraphx-driver; disabling bundled MiGraphX compiler tools.")
            set(AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER OFF)
        endif()
    endif()
endif()

if(AVE_INSTALL_BUNDLED_MODELS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

# ─────────────────────────────────────────────────────────────────
# FFmpeg (Required for Vulkan hardware frame extraction)
# ─────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libavfilter libswscale)

# ─────────────────────────────────────────────────────────────────
# Optional: libcurl
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_CURL)
    find_package(CURL QUIET)
    if(CURL_FOUND)
        message(STATUS "libcurl found: ${CURL_LIBRARIES}")
    else()
        message(WARNING "AVE_HAVE_CURL=ON but libcurl was not found; disabling.")
        set(AVE_HAVE_CURL OFF)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Optional: MiGraphX (ROCm)
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_MIGRAPHX)
    # MiGraphX depends on hiprtc; load it first.
    find_package(hiprtc CONFIG QUIET)
    find_package(migraphx CONFIG QUIET)
    if(NOT migraphx_FOUND)
        # Try pkg-config fallback
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(MIGRAPHX QUIET migraphx)
        endif()
        if(NOT MIGRAPHX_FOUND AND NOT migraphx_FOUND)
            message(WARNING "AVE_HAVE_MIGRAPHX=ON but MiGraphX not found; disabling.")
            set(AVE_HAVE_MIGRAPHX OFF)
        endif()
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Optional: HIP headers (supplied by ROCm)
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_HIP)
    find_package(hip CONFIG QUIET)
    if(NOT hip_FOUND)
        message(WARNING "AVE_HAVE_HIP=ON but HIP not found; disabling.")
        set(AVE_HAVE_HIP OFF)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Optional: ONNX Runtime ROCm/HIP fallback
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_ONNXRUNTIME_ROCM)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime/onnxruntime_cxx_api.h
        HINTS /usr/include /usr/local/include /opt/rocm/include)
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        HINTS /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/rocm/lib /opt/rocm/lib64)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(WARNING "AVE_HAVE_ONNXRUNTIME_ROCM=ON but ONNX Runtime was not found; disabling.")
        set(AVE_HAVE_ONNXRUNTIME_ROCM OFF)
    else()
        message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Optional: NCNN + Vulkan
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_NCNN)
    find_package(ncnn QUIET)
    if(NOT ncnn_FOUND)
        message(WARNING "AVE_HAVE_NCNN=ON but NCNN not found; disabling.")
        set(AVE_HAVE_NCNN OFF)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Optional: Vulkan (for Interop)
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_VULKAN)
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND)
        message(WARNING "AVE_HAVE_VULKAN=ON but Vulkan not found; disabling.")
        set(AVE_HAVE_VULKAN OFF)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Optional: ROCTx (part of roctracer; required for rocprof integration)
# ─────────────────────────────────────────────────────────────────
if(AVE_HAVE_ROCTX)
    find_path(ROCTX_INCLUDE_DIR  NAMES roctracer/roctx.h
              HINTS /opt/rocm/include)
    find_library(ROCTX_LIBRARY    NAMES roctx64
                 HINTS /opt/rocm/lib /opt/rocm/lib64)
    if(ROCTX_INCLUDE_DIR AND ROCTX_LIBRARY)
        message(STATUS "ROCTx found: ${ROCTX_LIBRARY}")
    else()
        message(WARNING "AVE_HAVE_ROCTX=ON but roctracer/roctx.h or libroctx64 not found; disabling.")
        set(AVE_HAVE_ROCTX OFF)
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Collect compile definitions
# ─────────────────────────────────────────────────────────────────
set(AVE_COMPILE_DEFS "")
if(AVE_HAVE_CURL)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_CURL)
endif()
if(AVE_HAVE_MIGRAPHX)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_MIGRAPHX)
endif()
if(AVE_HAVE_HIP)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_HIP)
endif()
if(AVE_HAVE_ONNXRUNTIME_ROCM)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_ONNXRUNTIME_ROCM)
endif()
if(AVE_HAVE_NCNN)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_NCNN)
endif()
if(AVE_HAVE_VULKAN)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_VULKAN)
endif()
if(AVE_HAVE_ROCTX)
    list(APPEND AVE_COMPILE_DEFS AVE_HAVE_ROCTX)
endif()

# ─────────────────────────────────────────────────────────────────
# Warning flags helper
# ─────────────────────────────────────────────────────────────────
function(ave_add_warnings target)
    if(AVE_STRICT_WARNINGS)
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /permissive-)
        else()
            target_compile_options(${target} PRIVATE
                -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
        endif()
    endif()
endfunction()

# ─────────────────────────────────────────────────────────────────
# Performance optimisation flags helper
# Enables native CPU ISA (AVX2, F16C, etc.) for the hot inference
# and pack/blit paths. Only applied to GCC/Clang on x86-family
# targets to keep cross-compilation and MSVC builds unaffected.
# ─────────────────────────────────────────────────────────────────
function(ave_add_native_optimizations target)
    if(NOT MSVC AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86"))
        target_compile_options(${target} PRIVATE -march=native -O3)
    endif()
endfunction()

function(ave_compute_prefix_rel_from_dir out_var install_dir)
    if(NOT IS_ABSOLUTE "${install_dir}")
        string(REPLACE "/" ";" _ave_dir_parts "${install_dir}")
        set(_ave_rel ".")
        foreach(_part IN LISTS _ave_dir_parts)
            if(_part STREQUAL "" OR _part STREQUAL ".")
                continue()
            endif()
            if(_ave_rel STREQUAL ".")
                set(_ave_rel "..")
            else()
                string(APPEND _ave_rel "/..")
            endif()
        endforeach()
    else()
        set(_ave_rel ".")
    endif()
    set(${out_var} "${_ave_rel}" PARENT_SCOPE)
endfunction()

function(ave_append_imported_target_locations out_var target_name)
    if(NOT TARGET ${target_name})
        return()
    endif()

    set(_ave_locations "")
    get_target_property(_ave_imported_configs ${target_name} IMPORTED_CONFIGURATIONS)
    foreach(_ave_config IN LISTS _ave_imported_configs)
        string(TOUPPER "${_ave_config}" _ave_config_upper)
        get_target_property(_ave_location ${target_name} "IMPORTED_LOCATION_${_ave_config_upper}")
        if(_ave_location AND EXISTS "${_ave_location}")
            list(APPEND _ave_locations "${_ave_location}")
        endif()
    endforeach()

    foreach(_ave_property IN ITEMS IMPORTED_LOCATION IMPORTED_IMPLIB)
        get_target_property(_ave_location ${target_name} ${_ave_property})
        if(_ave_location AND EXISTS "${_ave_location}")
            list(APPEND _ave_locations "${_ave_location}")
        endif()
    endforeach()

    if(_ave_locations)
        set(${out_var} "${${out_var}};${_ave_locations}" PARENT_SCOPE)
    endif()
endfunction()

# ─────────────────────────────────────────────────────────────────
# ave_core — static library
# ─────────────────────────────────────────────────────────────────
add_library(ave_core
    src/app_settings.cpp
    src/backend.cpp
    src/backend_manager.cpp
    src/cli.cpp
    src/error_taxonomy.cpp
    src/ffmpeg_runner.cpp
    src/filter_catalog.cpp
    src/frame_io.cpp
    src/frame_io_vulkan.cpp
    src/interop_bridge.cpp
    src/job_recovery.cpp
    src/job_queue.cpp
    src/model_catalog.cpp
    src/model_manager.cpp
    src/observability.cpp
    src/planner.cpp
    src/process_observer.cpp
    src/process_progress.cpp
    src/process_loop.cpp
    src/rgb_video_loop.cpp
    src/runtime_paths.cpp
    src/runtime_diagnostics.cpp
    src/scene_detector.cpp
    src/stage.cpp
    src/telemetry.cpp
    src/tensor_contract.cpp
    src/types.cpp
    src/video_probe.cpp
    src/video_processor.cpp
    src/vulkan_runtime.cpp
    src/backends/glsl_shader_backend.cpp
    src/backends/migraphx_backend.cpp
    src/backends/ncnn_vulkan_backend.cpp
    src/backends/rocm_hip_backend.cpp
    src/backends/vapoursynth_backend.cpp
    src/backends/vulkan_compute_backend.cpp
)

target_include_directories(ave_core PUBLIC include ${FFMPEG_INCLUDE_DIRS})
target_compile_definitions(ave_core PUBLIC ${AVE_COMPILE_DEFS})
ave_add_warnings(ave_core)
ave_add_native_optimizations(ave_core)

# Link optional libraries to ave_core
target_link_libraries(ave_core PUBLIC ${FFMPEG_LIBRARIES})
if(AVE_HAVE_CURL AND CURL_FOUND)
    target_link_libraries(ave_core PUBLIC CURL::libcurl)
endif()
if(AVE_HAVE_MIGRAPHX)
    if(migraphx_FOUND)
        target_link_libraries(ave_core PUBLIC migraphx::c migraphx::gpu)
    elseif(MIGRAPHX_FOUND)
        target_include_directories(ave_core PUBLIC ${MIGRAPHX_INCLUDE_DIRS})
        target_link_libraries(ave_core PUBLIC ${MIGRAPHX_LIBRARIES})
    endif()
endif()
if(AVE_HAVE_HIP AND hip_FOUND)
    target_link_libraries(ave_core PUBLIC hip::host)
endif()
if(AVE_HAVE_ONNXRUNTIME_ROCM)
    target_include_directories(ave_core PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(ave_core PUBLIC ${ONNXRUNTIME_LIBRARY})
endif()
if(AVE_HAVE_NCNN AND ncnn_FOUND)
    target_link_libraries(ave_core PUBLIC ncnn)
endif()
if(AVE_HAVE_VULKAN AND Vulkan_FOUND)
    target_link_libraries(ave_core PUBLIC Vulkan::Vulkan)
endif()
if(AVE_HAVE_ROCTX)
    target_include_directories(ave_core PUBLIC ${ROCTX_INCLUDE_DIR})
    target_link_libraries(ave_core PUBLIC ${ROCTX_LIBRARY})
endif()

set(AVE_RUNTIME_BUNDLE_SEED_LIBS "")
set(AVE_RUNTIME_BUNDLE_SEED_EXECUTABLES "")
if(AVE_HAVE_MIGRAPHX)
    if(migraphx_FOUND)
        ave_append_imported_target_locations(AVE_RUNTIME_BUNDLE_SEED_LIBS migraphx::c)
        ave_append_imported_target_locations(AVE_RUNTIME_BUNDLE_SEED_LIBS migraphx::gpu)
    endif()

    if(MIGRAPHX_FOUND)
        set(_AVE_MIGRAPHX_RUNTIME_SEARCH_DIRS ${MIGRAPHX_LIBRARY_DIRS})
        foreach(_ave_dir IN LISTS MIGRAPHX_LIBRARY_DIRS)
            if(EXISTS "${_ave_dir}/migraphx/lib")
                list(APPEND _AVE_MIGRAPHX_RUNTIME_SEARCH_DIRS "${_ave_dir}/migraphx/lib")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _AVE_MIGRAPHX_RUNTIME_SEARCH_DIRS)

        foreach(_ave_dir IN LISTS _AVE_MIGRAPHX_RUNTIME_SEARCH_DIRS)
            foreach(_ave_pattern IN ITEMS
                    libmigraphx_c.so*
                    libmigraphx.so*
                    libmigraphx_gpu.so*
                    libmigraphx_device.so*
                    libmigraphx_ref.so*)
                file(GLOB _ave_seed_matches
                    LIST_DIRECTORIES false
                    "${_ave_dir}/${_ave_pattern}")
                list(APPEND AVE_RUNTIME_BUNDLE_SEED_LIBS ${_ave_seed_matches})
            endforeach()
        endforeach()
    endif()

    list(REMOVE_DUPLICATES AVE_RUNTIME_BUNDLE_SEED_LIBS)
endif()

find_program(AVE_GLSL_COMPILER_EXECUTABLE
    NAMES glslc glslangValidator)
if(AVE_GLSL_COMPILER_EXECUTABLE)
    list(APPEND AVE_RUNTIME_BUNDLE_SEED_EXECUTABLES
        "${AVE_GLSL_COMPILER_EXECUTABLE}")
endif()
list(REMOVE_DUPLICATES AVE_RUNTIME_BUNDLE_SEED_EXECUTABLES)

find_program(AVE_FFMPEG_CLI_EXECUTABLE NAMES ffmpeg)
find_program(AVE_FFPROBE_CLI_EXECUTABLE NAMES ffprobe)
foreach(_ave_media_tool IN LISTS AVE_FFMPEG_CLI_EXECUTABLE AVE_FFPROBE_CLI_EXECUTABLE)
    if(_ave_media_tool)
        list(APPEND AVE_RUNTIME_BUNDLE_SEED_EXECUTABLES "${_ave_media_tool}")
    endif()
endforeach()
list(REMOVE_DUPLICATES AVE_RUNTIME_BUNDLE_SEED_EXECUTABLES)

# ─────────────────────────────────────────────────────────────────
# ave — CLI executable
# ─────────────────────────────────────────────────────────────────
add_executable(ave src/main.cpp)
target_link_libraries(ave PRIVATE ave_core)
ave_add_warnings(ave)

set(_AVE_INSTALL_RPATH
    "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}/ave/runtime;$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}/ave/migraphx/lib;$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}/ave/migraphx/lib/migraphx/lib")
set_target_properties(ave PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "${_AVE_INSTALL_RPATH}")

add_executable(ave_model_compile_sweep
    tools/compile_onnx_to_mxr.cpp
)
target_link_libraries(ave_model_compile_sweep PRIVATE ave_core)
ave_add_warnings(ave_model_compile_sweep)
ave_add_native_optimizations(ave_model_compile_sweep)
set_target_properties(ave_model_compile_sweep PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "${_AVE_INSTALL_RPATH}")

# ─────────────────────────────────────────────────────────────────
# ave_gui — Qt desktop GUI
# ─────────────────────────────────────────────────────────────────
if(AVE_BUILD_GUI)
    find_package(Qt6 COMPONENTS Widgets QUIET)
    if(Qt6Widgets_FOUND)
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTOUIC ON)
        set(CMAKE_AUTORCC ON)

        add_executable(ave_gui
            src/gui/main_gui.cpp
            src/gui/ave_gui_resources.qrc
            src/gui/main_window.cpp
            src/gui/main_window.hpp
            src/gui/model_manager_dialog.cpp
            src/gui/model_manager_dialog.hpp
            src/gui/settings_dialog.cpp
            src/gui/settings_dialog.hpp
            src/gui/toggle_switch.cpp
            src/gui/toggle_switch.hpp
            src/gui/filter_browser.cpp
            src/gui/filter_browser.hpp
        )
        target_include_directories(ave_gui PRIVATE src/gui)
        target_link_libraries(ave_gui PRIVATE ave_core Qt6::Widgets)
        target_compile_definitions(ave_gui PRIVATE ${AVE_COMPILE_DEFS})
        ave_add_warnings(ave_gui)
        set_target_properties(ave_gui PROPERTIES
            BUILD_RPATH_USE_ORIGIN ON
            INSTALL_RPATH "${_AVE_INSTALL_RPATH}")
        message(STATUS "Qt6 Widgets found — building ave_gui")
    else()
        message(STATUS "Qt6 Widgets not found — skipping ave_gui target")
    endif()
endif()

# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────
enable_testing()

add_executable(planner_tests
    tests/planner_tests.cpp
    src/model_catalog.cpp
    src/planner.cpp
    src/stage.cpp
    src/types.cpp
)
target_include_directories(planner_tests PRIVATE include)
target_compile_definitions(planner_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(planner_tests)
add_test(NAME planner_tests COMMAND planner_tests)

add_executable(tensor_contract_tests
    tests/tensor_contract_tests.cpp
    src/tensor_contract.cpp
)
target_include_directories(tensor_contract_tests PRIVATE include)
target_compile_definitions(tensor_contract_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(tensor_contract_tests)
add_test(NAME tensor_contract_tests COMMAND tensor_contract_tests)

add_executable(runtime_diagnostics_tests
    tests/runtime_diagnostics_tests.cpp
    src/process_observer.cpp
    src/runtime_diagnostics.cpp
    src/runtime_paths.cpp
    src/video_probe.cpp
)
target_include_directories(runtime_diagnostics_tests PRIVATE include)
target_compile_definitions(runtime_diagnostics_tests PRIVATE ${AVE_COMPILE_DEFS})
if(AVE_HAVE_HIP AND hip_FOUND)
    target_link_libraries(runtime_diagnostics_tests PRIVATE hip::host)
endif()
if(AVE_HAVE_ONNXRUNTIME_ROCM)
    target_include_directories(runtime_diagnostics_tests PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(runtime_diagnostics_tests PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()
ave_add_warnings(runtime_diagnostics_tests)
add_test(NAME runtime_diagnostics_tests COMMAND runtime_diagnostics_tests)

add_executable(runtime_paths_tests
    tests/runtime_paths_tests.cpp
    src/runtime_paths.cpp
)
target_include_directories(runtime_paths_tests PRIVATE include)
target_compile_definitions(runtime_paths_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(runtime_paths_tests)
add_test(NAME runtime_paths_tests COMMAND runtime_paths_tests)

add_executable(model_manager_profile_tests
    tests/model_manager_profile_tests.cpp
)
target_link_libraries(model_manager_profile_tests PRIVATE ave_core)
target_include_directories(model_manager_profile_tests PRIVATE include)
target_compile_definitions(model_manager_profile_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(model_manager_profile_tests)
add_test(NAME model_manager_profile_tests COMMAND model_manager_profile_tests)

add_executable(video_probe_tests
    tests/video_probe_tests.cpp
    src/process_observer.cpp
    src/video_probe.cpp
)
target_include_directories(video_probe_tests PRIVATE include)
target_compile_definitions(video_probe_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(video_probe_tests)
add_test(NAME video_probe_tests COMMAND video_probe_tests)

add_executable(process_loop_tests
    tests/process_loop_tests.cpp
    src/process_loop.cpp
)
target_include_directories(process_loop_tests PRIVATE include)
target_compile_definitions(process_loop_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(process_loop_tests)
add_test(NAME process_loop_tests COMMAND process_loop_tests)

add_executable(process_observer_tests
    tests/process_observer_tests.cpp
    src/process_observer.cpp
    src/video_probe.cpp
)
target_include_directories(process_observer_tests PRIVATE include)
target_compile_definitions(process_observer_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(process_observer_tests)
add_test(NAME process_observer_tests COMMAND process_observer_tests)

add_executable(process_progress_tests
    tests/process_progress_tests.cpp
    src/process_observer.cpp
    src/process_progress.cpp
    src/video_probe.cpp
)
target_include_directories(process_progress_tests PRIVATE include)
target_compile_definitions(process_progress_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(process_progress_tests)
add_test(NAME process_progress_tests COMMAND process_progress_tests)

add_executable(rgb_video_loop_tests
    tests/rgb_video_loop_tests.cpp
)
target_link_libraries(rgb_video_loop_tests PRIVATE ave_core)
target_include_directories(rgb_video_loop_tests PRIVATE include)
target_compile_definitions(rgb_video_loop_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(rgb_video_loop_tests)
add_test(NAME rgb_video_loop_tests COMMAND rgb_video_loop_tests)

add_executable(frame_io_vulkan_tests
    tests/frame_io_vulkan_tests.cpp
)
target_link_libraries(frame_io_vulkan_tests PRIVATE ave_core)
target_include_directories(frame_io_vulkan_tests PRIVATE include)
target_compile_definitions(frame_io_vulkan_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(frame_io_vulkan_tests)
add_test(NAME frame_io_vulkan_tests COMMAND frame_io_vulkan_tests)

add_executable(frame_io_pipe_tests
    tests/frame_io_pipe_tests.cpp
)
target_link_libraries(frame_io_pipe_tests PRIVATE ave_core)
target_include_directories(frame_io_pipe_tests PRIVATE include)
target_compile_definitions(frame_io_pipe_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(frame_io_pipe_tests)
add_test(NAME frame_io_pipe_tests COMMAND frame_io_pipe_tests)

add_executable(frame_io_direct_pack_tests
    tests/frame_io_direct_pack_tests.cpp
)
target_link_libraries(frame_io_direct_pack_tests PRIVATE ave_core)
target_include_directories(frame_io_direct_pack_tests PRIVATE include)
target_compile_definitions(frame_io_direct_pack_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(frame_io_direct_pack_tests)
add_test(NAME frame_io_direct_pack_tests COMMAND frame_io_direct_pack_tests)

add_executable(ffmpeg_runner_tests
    tests/ffmpeg_runner_tests.cpp
)
target_link_libraries(ffmpeg_runner_tests PRIVATE ave_core)
target_include_directories(ffmpeg_runner_tests PRIVATE include)
target_compile_definitions(ffmpeg_runner_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(ffmpeg_runner_tests)
add_test(NAME ffmpeg_runner_tests COMMAND ffmpeg_runner_tests)

add_executable(job_recovery_tests
    tests/job_recovery_tests.cpp
    src/job_recovery.cpp
    src/stage.cpp
    src/types.cpp
    src/filter_catalog.cpp
    src/backend.cpp
)
target_include_directories(job_recovery_tests PRIVATE include)
target_compile_definitions(job_recovery_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(job_recovery_tests)
add_test(NAME job_recovery_tests COMMAND job_recovery_tests)

add_executable(job_queue_tests
    tests/job_queue_tests.cpp
    src/job_queue.cpp
    src/app_settings.cpp
    src/backend.cpp
    src/filter_catalog.cpp
    src/stage.cpp
    src/types.cpp
)
target_include_directories(job_queue_tests PRIVATE include)
target_compile_definitions(job_queue_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(job_queue_tests)
add_test(NAME job_queue_tests COMMAND job_queue_tests)

add_executable(telemetry_tests
    tests/telemetry_tests.cpp
    src/process_observer.cpp
    src/telemetry.cpp
    src/video_probe.cpp
)
target_include_directories(telemetry_tests PRIVATE include)
target_compile_definitions(telemetry_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(telemetry_tests)
add_test(NAME telemetry_tests COMMAND telemetry_tests)

add_executable(migraphx_backend_tests
    tests/migraphx_backend_tests.cpp
)
target_link_libraries(migraphx_backend_tests PRIVATE ave_core)
target_include_directories(migraphx_backend_tests PRIVATE include)
target_compile_definitions(migraphx_backend_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(migraphx_backend_tests)
add_test(NAME migraphx_backend_tests COMMAND migraphx_backend_tests)

add_executable(interop_bridge_tests
    tests/interop_bridge_tests.cpp
)
target_link_libraries(interop_bridge_tests PRIVATE ave_core)
target_include_directories(interop_bridge_tests PRIVATE include)
target_compile_definitions(interop_bridge_tests PRIVATE ${AVE_COMPILE_DEFS})
ave_add_warnings(interop_bridge_tests)
add_test(NAME interop_bridge_tests COMMAND interop_bridge_tests)

set(AVE_INSTALL_DOC_DIR "${CMAKE_INSTALL_DATADIR}/doc/amd-video-enhancer")
set(AVE_INSTALL_LICENSE_DIR "${CMAKE_INSTALL_DATADIR}/licenses/amd-video-enhancer")

install(FILES
    README.md
    CHANGELOG.md
    SECURITY.md
    docs/BENCHMARKS.md
    docs/BETA_TESTING_PROGRAM.md
    docs/LIMITATIONS.md
    docs/PACKAGING.md
    docs/RELEASE_STATUS.md
    docs/RELEASING.md
    docs/SUPPORT_TIERS.md
    docs/VALIDATION_AND_EVIDENCE.md
    DESTINATION ${AVE_INSTALL_DOC_DIR}
)

install(FILES
    LICENSE
    DESTINATION ${AVE_INSTALL_LICENSE_DIR}
)

install(TARGETS ave
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ave
)
install(TARGETS ave_model_compile_sweep
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ave
)

if(TARGET ave_gui)
    install(TARGETS ave_gui
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ave
    )
endif()

set(AVE_INSTALL_HAS_GUI_TARGET FALSE)
if(TARGET ave_gui)
    set(AVE_INSTALL_HAS_GUI_TARGET TRUE)
endif()

ave_compute_prefix_rel_from_dir(AVE_PREFIX_REL_FROM_BINDIR "${CMAKE_INSTALL_BINDIR}")
set(AVE_INSTALL_MIGRAPHX_BINDIR "${CMAKE_INSTALL_LIBDIR}/ave/migraphx/bin")
ave_compute_prefix_rel_from_dir(AVE_PREFIX_REL_FROM_MIGRAPHX_BINDIR "${AVE_INSTALL_MIGRAPHX_BINDIR}")

set(AVE_INSTALL_APP_BINARY_PATH "${CMAKE_INSTALL_LIBEXECDIR}/ave/ave")
set(AVE_INSTALL_MIGRAPHX_PATH "${CMAKE_INSTALL_LIBDIR}/ave/migraphx")
set(AVE_INSTALL_RUNTIME_PATH "${CMAKE_INSTALL_LIBDIR}/ave/runtime")
set(AVE_INSTALL_QT_PLUGIN_PATH "${CMAKE_INSTALL_LIBDIR}/ave/qt-plugins")
set(AVE_INSTALL_MODELS_PATH "${CMAKE_INSTALL_DATADIR}/ave/models")
set(AVE_INSTALL_TOOLS_PATH "${CMAKE_INSTALL_LIBEXECDIR}/ave/tools")
configure_file(cmake/ave-launch.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/ave-launch-wrapper.sh
               @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ave-launch-wrapper.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        RENAME ave)

set(AVE_INSTALL_APP_BINARY_PATH "${CMAKE_INSTALL_LIBEXECDIR}/ave/ave_model_compile_sweep")
configure_file(cmake/ave-launch.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/ave-model-compile-sweep-wrapper.sh
               @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ave-model-compile-sweep-wrapper.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        RENAME ave_model_compile_sweep)

if(TARGET ave_gui)
    set(AVE_INSTALL_APP_BINARY_PATH "${CMAKE_INSTALL_LIBEXECDIR}/ave/ave_gui")
    configure_file(cmake/ave-launch.sh.in
                   ${CMAKE_CURRENT_BINARY_DIR}/ave-gui-launch-wrapper.sh
                   @ONLY)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ave-gui-launch-wrapper.sh
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            RENAME ave_gui)
endif()

if(AVE_INSTALL_BUNDLED_MIGRAPHX)
    install(DIRECTORY
        "${AVE_BUNDLED_MIGRAPHX_PREFIX}/lib/migraphx/lib"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/lib/migraphx
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING
        PATTERN "libmigraphx.so*"
        PATTERN "libmigraphx_gpu.so*"
        PATTERN "libmigraphx_device.so*"
        PATTERN "libmigraphx_ref.so*"
    )
    if(AVE_INSTALL_BUNDLED_MIGRAPHX_COMPILER)
        install(PROGRAMS
            "${AVE_BUNDLED_MIGRAPHX_PREFIX}/bin/migraphx-driver"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/bin
        )
        if(EXISTS "${AVE_BUNDLED_MIGRAPHX_PREFIX}/bin/migraphx-hiprtc-driver")
            install(PROGRAMS
                "${AVE_BUNDLED_MIGRAPHX_PREFIX}/bin/migraphx-hiprtc-driver"
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/bin
            )
        endif()
        configure_file(cmake/ave-migraphx-driver.sh.in
                       ${CMAKE_CURRENT_BINARY_DIR}/ave-migraphx-driver-wrapper.sh
                       @ONLY)
        install(PROGRAMS
            ${CMAKE_CURRENT_BINARY_DIR}/ave-migraphx-driver-wrapper.sh
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/bin
            RENAME ave-migraphx-driver
        )
        install(DIRECTORY
            "${AVE_BUNDLED_MIGRAPHX_PREFIX}/lib/migraphx/lib"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/lib/migraphx
            USE_SOURCE_PERMISSIONS
            FILES_MATCHING
            PATTERN "libmigraphx_onnx.so*"
            PATTERN "libmigraphx_py.so*"
            PATTERN "libmigraphx_tf.so*"
        )
    endif()
    configure_file(cmake/install_bundled_migraphx_deps.cmake.in
                   ${CMAKE_CURRENT_BINARY_DIR}/install_bundled_migraphx_deps.cmake
                   @ONLY)
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_bundled_migraphx_deps.cmake)
endif()

set(AVE_INSTALL_TOOL_EXECUTABLE_PATHS "")
if(AVE_GLSL_COMPILER_EXECUTABLE)
    get_filename_component(_ave_glsl_compiler_name "${AVE_GLSL_COMPILER_EXECUTABLE}" NAME)
    install(PROGRAMS
        "${AVE_GLSL_COMPILER_EXECUTABLE}"
        DESTINATION ${AVE_INSTALL_TOOLS_PATH})
    list(APPEND AVE_INSTALL_TOOL_EXECUTABLE_PATHS
        "${AVE_INSTALL_TOOLS_PATH}/${_ave_glsl_compiler_name}")
endif()
foreach(_ave_media_tool IN ITEMS "${AVE_FFMPEG_CLI_EXECUTABLE}" "${AVE_FFPROBE_CLI_EXECUTABLE}")
    if(_ave_media_tool)
        get_filename_component(_ave_media_tool_name "${_ave_media_tool}" NAME)
        install(PROGRAMS
            "${_ave_media_tool}"
            DESTINATION ${AVE_INSTALL_TOOLS_PATH})
        list(APPEND AVE_INSTALL_TOOL_EXECUTABLE_PATHS
            "${AVE_INSTALL_TOOLS_PATH}/${_ave_media_tool_name}")
    endif()
endforeach()
list(REMOVE_DUPLICATES AVE_INSTALL_TOOL_EXECUTABLE_PATHS)

if(AVE_INSTALL_BUNDLED_MODELS)
    configure_file(cmake/install_model_bundle.cmake.in
                   ${CMAKE_CURRENT_BINARY_DIR}/install_model_bundle.cmake
                   @ONLY)
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_model_bundle.cmake)
endif()

if(AVE_INSTALL_RUNTIME_DEPS)
    configure_file(cmake/install_runtime_deps.cmake.in
                   ${CMAKE_CURRENT_BINARY_DIR}/install_runtime_deps.cmake
                   @ONLY)
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_runtime_deps.cmake)
endif()

if(TARGET ave_gui AND Qt6Widgets_FOUND)
    set(AVE_QT_PLUGINS_DIR "")
    set(AVE_QT_PLUGIN_SUBDIRS
        platforms
        platformthemes
        platforminputcontexts
        styles
        imageformats
        iconengines
        tls
        xcbglintegrations
        egldeviceintegrations
        wayland-decoration-client
        wayland-graphics-integration-client
        wayland-shell-integration)
    set(AVE_QT_PLUGIN_EXCLUDED_GLOBS
        imageformats/kimg_jxr.so
        imageformats/kimg_jxr.so.*)
    if(TARGET Qt6::qmake)
        get_target_property(AVE_QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
        if(AVE_QT_QMAKE_EXECUTABLE AND EXISTS "${AVE_QT_QMAKE_EXECUTABLE}")
            execute_process(
                COMMAND "${AVE_QT_QMAKE_EXECUTABLE}" -query QT_INSTALL_PLUGINS
                OUTPUT_VARIABLE AVE_QT_PLUGINS_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
        endif()
    endif()
    if(NOT AVE_QT_PLUGINS_DIR STREQUAL "")
        configure_file(cmake/install_qt_plugins.cmake.in
                       ${CMAKE_CURRENT_BINARY_DIR}/install_qt_plugins.cmake
                       @ONLY)
        install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_qt_plugins.cmake)
    else()
        message(WARNING "Qt GUI is enabled but the Qt plugin directory could not be resolved; portable GUI bundles may miss Qt platform plugins.")
    endif()
endif()

if(NOT Python3_Interpreter_FOUND)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

set(AVE_PORTABLE_BUNDLE_NAME "amd-video-enhancer-portable-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
    CACHE STRING "Folder/archive name for the portable bundle output")
set(AVE_PORTABLE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/dist"
    CACHE PATH "Output directory for portable bundle staging and archives")

set(_AVE_PORTABLE_TARGET_DEPS ave ave_model_compile_sweep)
if(TARGET ave_gui)
    list(APPEND _AVE_PORTABLE_TARGET_DEPS ave_gui)
endif()

add_custom_target(portable_stage
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_portable_bundle.py"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --output-dir "${AVE_PORTABLE_OUTPUT_DIR}"
            --bundle-name "${AVE_PORTABLE_BUNDLE_NAME}"
            --skip-archive
    DEPENDS ${_AVE_PORTABLE_TARGET_DEPS}
    COMMENT "Staging a self-contained portable AMD Video Enhancer bundle")

add_custom_target(portable_bundle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_portable_bundle.py"
            --build-dir "${CMAKE_CURRENT_BINARY_DIR}"
            --output-dir "${AVE_PORTABLE_OUTPUT_DIR}"
            --bundle-name "${AVE_PORTABLE_BUNDLE_NAME}"
    DEPENDS portable_stage
    COMMENT "Creating a compressed portable AMD Video Enhancer bundle")
