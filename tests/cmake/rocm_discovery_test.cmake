cmake_minimum_required(VERSION 3.21)

set(_ave_test_root "${CMAKE_CURRENT_BINARY_DIR}/fake-rocm")
file(REMOVE_RECURSE "${_ave_test_root}")
file(MAKE_DIRECTORY "${_ave_test_root}/lib/cmake/migraphx")

file(WRITE "${_ave_test_root}/lib/cmake/migraphx/migraphxConfig.cmake"
    "set(migraphx_FOUND TRUE)\n")

set(AVE_ROCM_ROOT_HINTS "${_ave_test_root}")
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/ave_rocm_discovery.cmake")
ave_discover_rocm_prefixes()

list(FIND CMAKE_PREFIX_PATH "${_ave_test_root}" _ave_prefix_index)
if(_ave_prefix_index EQUAL -1)
    message(FATAL_ERROR "ROCm discovery did not add the explicit test prefix to CMAKE_PREFIX_PATH")
endif()

find_package(migraphx CONFIG QUIET)
if(NOT migraphx_FOUND)
    message(FATAL_ERROR "CMake could not discover a CONFIG package underneath the discovered ROCm prefix")
endif()

message(STATUS "ROCm prefix discovery regression test passed")
