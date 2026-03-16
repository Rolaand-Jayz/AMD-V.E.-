cmake_minimum_required(VERSION 3.21)

project(amd_video_enhancer_cpp LANGUAGES CXX)

include(GNUInstallDirs)

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
option(AVE_HAVE_NCNN         "Enable NCNN Vulkan runtime"            OFF)
option(AVE_HAVE_VULKAN       "Enable Vulkan runtime for interop"     ON)
# ROCTx: zero-overhead stage markers for rocprof --roctx-trace integration.
# Requires roctracer-dev package: sudo apt install roctracer-dev
option(AVE_HAVE_ROCTX        "Enable ROCTx markers for rocprof"      OFF)
option(AVE_INSTALL_BUNDLED_MIGRAPHX
       "Install a bundled patched MiGraphX runtime with the app"     ON)
option(AVE_INSTALL_BUNDLED_MODELS
       "Install the full model catalog with the app"                 ON)

set(_AVE_DEFAULT_BUNDLED_MIGRAPHX_PREFIX "")
if(EXISTS "$ENV{HOME}/.local/opt/migraphx-codex/bin/migraphx-driver")
    set(_AVE_DEFAULT_BUNDLED_MIGRAPHX_PREFIX "$ENV{HOME}/.local/opt/migraphx-codex")
endif()
set(AVE_BUNDLED_MIGRAPHX_PREFIX "${_AVE_DEFAULT_BUNDLED_MIGRAPHX_PREFIX}" CACHE PATH
    "Patched MiGraphX install prefix to bundle with the app")

if(AVE_INSTALL_BUNDLED_MIGRAPHX)
    if(AVE_BUNDLED_MIGRAPHX_PREFIX STREQUAL "" OR
       NOT EXISTS "${AVE_BUNDLED_MIGRAPHX_PREFIX}/bin/migraphx-driver")
        message(WARNING
            "AVE_INSTALL_BUNDLED_MIGRAPHX=ON but AVE_BUNDLED_MIGRAPHX_PREFIX does not contain "
            "bin/migraphx-driver; disabling bundled MiGraphX install.")
        set(AVE_INSTALL_BUNDLED_MIGRAPHX OFF)
    else()
        message(STATUS "Bundling patched MiGraphX runtime from ${AVE_BUNDLED_MIGRAPHX_PREFIX}")
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
    find_package(hiprtc QUIET)
    find_package(migraphx QUIET)
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
    find_package(hip QUIET)
    if(NOT hip_FOUND)
        message(WARNING "AVE_HAVE_HIP=ON but HIP not found; disabling.")
        set(AVE_HAVE_HIP OFF)
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
    src/model_catalog.cpp
    src/model_manager.cpp
    src/observability.cpp
    src/planner.cpp
    src/runtime_paths.cpp
    src/scene_detector.cpp
    src/stage.cpp
    src/tensor_contract.cpp
    src/types.cpp
    src/video_processor.cpp
    src/vulkan_runtime.cpp
    src/backends/glsl_shader_backend.cpp
    src/backends/migraphx_backend.cpp
    src/backends/ncnn_vulkan_backend.cpp
    src/backends/vapoursynth_backend.cpp
    src/backends/vulkan_compute_backend.cpp
)

target_include_directories(ave_core PUBLIC include ${FFMPEG_INCLUDE_DIRS})
target_compile_definitions(ave_core PUBLIC ${AVE_COMPILE_DEFS})
ave_add_warnings(ave_core)

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

# ─────────────────────────────────────────────────────────────────
# ave — CLI executable
# ─────────────────────────────────────────────────────────────────
add_executable(ave src/main.cpp)
target_link_libraries(ave PRIVATE ave_core)
ave_add_warnings(ave)

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

install(TARGETS ave
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ave
)

if(TARGET ave_gui)
    install(TARGETS ave_gui
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/ave
    )
endif()

if(NOT IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}")
    string(REPLACE "/" ";" _ave_bindir_parts "${CMAKE_INSTALL_BINDIR}")
    set(AVE_PREFIX_REL_FROM_BINDIR ".")
    foreach(_part IN LISTS _ave_bindir_parts)
        if(_part STREQUAL "" OR _part STREQUAL ".")
            continue()
        endif()
        if(AVE_PREFIX_REL_FROM_BINDIR STREQUAL ".")
            set(AVE_PREFIX_REL_FROM_BINDIR "..")
        else()
            string(APPEND AVE_PREFIX_REL_FROM_BINDIR "/..")
        endif()
    endforeach()
else()
    set(AVE_PREFIX_REL_FROM_BINDIR ".")
endif()

set(AVE_INSTALL_APP_BINARY_PATH "${CMAKE_INSTALL_LIBEXECDIR}/ave/ave")
set(AVE_INSTALL_MIGRAPHX_PATH "${CMAKE_INSTALL_LIBDIR}/ave/migraphx")
set(AVE_INSTALL_MODELS_PATH "${CMAKE_INSTALL_DATADIR}/ave/models")
configure_file(cmake/ave-launch.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/ave-launch-wrapper.sh
               @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ave-launch-wrapper.sh
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        RENAME ave)

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
    install(PROGRAMS
        "${AVE_BUNDLED_MIGRAPHX_PREFIX}/bin/migraphx-driver"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/bin
    )
    install(DIRECTORY
        "${AVE_BUNDLED_MIGRAPHX_PREFIX}/lib/migraphx/lib"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/lib/migraphx
        USE_SOURCE_PERMISSIONS
    )
    install(DIRECTORY
        "${AVE_BUNDLED_MIGRAPHX_PREFIX}/lib/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/ave/migraphx/lib
        USE_SOURCE_PERMISSIONS
        FILES_MATCHING
        PATTERN "cmake" EXCLUDE
        PATTERN "migraphx" EXCLUDE
        PATTERN "libmigraphx*.so*"
    )
endif()

if(AVE_INSTALL_BUNDLED_MODELS)
    configure_file(cmake/install_model_bundle.cmake.in
                   ${CMAKE_CURRENT_BINARY_DIR}/install_model_bundle.cmake
                   @ONLY)
    install(SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/install_model_bundle.cmake)
endif()
