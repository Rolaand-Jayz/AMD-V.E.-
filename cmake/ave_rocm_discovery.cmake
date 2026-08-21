# ROCm CMake package discovery helpers.
#
# ROCm packages predominantly use CMake CONFIG packages installed below a
# common prefix (normally /opt/rocm or /opt/rocm-<version>).  CMake does not
# automatically search arbitrary /opt prefixes, so seed CMAKE_PREFIX_PATH from
# the standard ROCm locations and the environment before find_package() calls.

function(ave_discover_rocm_prefixes)
    set(_ave_rocm_candidates "")

    # Explicit project hints take priority and make non-standard/test installs
    # deterministic. This may be a semicolon-separated list.
    if(DEFINED AVE_ROCM_ROOT_HINTS AND NOT AVE_ROCM_ROOT_HINTS STREQUAL "")
        list(APPEND _ave_rocm_candidates ${AVE_ROCM_ROOT_HINTS})
    endif()

    # Common ROCm environment variables used by packaged and custom installs.
    foreach(_ave_env_var IN ITEMS ROCM_PATH ROCM_HOME)
        if(DEFINED ENV{${_ave_env_var}} AND NOT "$ENV{${_ave_env_var}}" STREQUAL "")
            list(APPEND _ave_rocm_candidates "$ENV{${_ave_env_var}}")
        endif()
    endforeach()

    # HIP_PATH is normally the ROCm root on current installs, but older layouts
    # may point at <rocm>/hip. Consider both without assuming either layout.
    if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        set(_ave_hip_path "$ENV{HIP_PATH}")
        list(APPEND _ave_rocm_candidates "${_ave_hip_path}")
        get_filename_component(_ave_hip_leaf "${_ave_hip_path}" NAME)
        if(_ave_hip_leaf STREQUAL "hip")
            get_filename_component(_ave_hip_parent "${_ave_hip_path}" DIRECTORY)
            list(APPEND _ave_rocm_candidates "${_ave_hip_parent}")
        endif()
    endif()

    # AMD packages normally expose /opt/rocm, while side-by-side installs can
    # retain versioned roots such as /opt/rocm-6.2.2.
    list(APPEND _ave_rocm_candidates /opt/rocm)
    file(GLOB _ave_versioned_rocm_roots LIST_DIRECTORIES true "/opt/rocm-*")
    list(APPEND _ave_rocm_candidates ${_ave_versioned_rocm_roots})

    list(REMOVE_DUPLICATES _ave_rocm_candidates)

    set(_ave_rocm_found "")
    foreach(_ave_rocm_root IN LISTS _ave_rocm_candidates)
        if(IS_DIRECTORY "${_ave_rocm_root}")
            list(APPEND _ave_rocm_found "${_ave_rocm_root}")
            list(FIND CMAKE_PREFIX_PATH "${_ave_rocm_root}" _ave_prefix_index)
            if(_ave_prefix_index EQUAL -1)
                list(APPEND CMAKE_PREFIX_PATH "${_ave_rocm_root}")
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
    list(REMOVE_DUPLICATES _ave_rocm_found)

    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
    set(AVE_ROCM_DISCOVERED_PREFIXES "${_ave_rocm_found}" PARENT_SCOPE)
endfunction()
