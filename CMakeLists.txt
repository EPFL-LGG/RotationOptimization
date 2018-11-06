################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.1)
project(rotation_optimization)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
    )

if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

# Enable more warnings
include(Warnings)

# Make sure warnings/errors are still colorized when using Ninja for building.
add_definitions(-fdiagnostics-color=always)

# Export compile flags(used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Common libraries
################################################################################
include(DownloadProject)
set(ROTOPT_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(ROTOPT_EXTERNAL "${ROTOPT_ROOT}/3rdparty")

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
    set(ROTOPT_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
    set(ROTOPT_EXTRA_OPTIONS "")
endif()


function(rotopt_download_project name)
    download_project(
        PROJ         ${name}
        SOURCE_DIR   ${ROTOPT_EXTERNAL}/${name}
        DOWNLOAD_DIR ${ROTOPT_EXTERNAL}/.cache/${name}
        QUIET
        ${ROTOPT_EXTRA_OPTIONS}
        ${ARGN}
    )
endfunction()

# Eigen3 library
if(NOT TARGET Eigen3::Eigen)
    add_library(rotopt_eigen INTERFACE)
    rotopt_download_project(eigen
        URL     http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
        URL_MD5 a7aab9f758249b86c93221ad417fbe18
    )
    target_include_directories(rotopt_eigen SYSTEM INTERFACE ${ROTOPT_EXTERNAL}/eigen)
    add_library(Eigen3::Eigen ALIAS rotopt_eigen)
endif()

################################################################################
# Rotation optimization library
################################################################################
add_library(rotation_optimization rotation_optimization.cc)
target_link_libraries(rotation_optimization PUBLIC Eigen3::Eigen)
target_include_directories(rotation_optimization PUBLIC .)
set_target_properties(rotation_optimization PROPERTIES CXX_STANDARD 11)
set_target_properties(rotation_optimization PROPERTIES CXX_STANDARD_REQUIRED ON)

################################################################################
# Unit tests
################################################################################
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Catch2
    if(NOT TARGET Catch2::Catch2)
        rotopt_download_project(Catch2
            URL     https://github.com/catchorg/Catch2/archive/v2.3.0.tar.gz
            URL_MD5 1fc90ff3b7b407b83057537f4136489e
        )
        add_subdirectory(${ROTOPT_EXTERNAL}/Catch2)
        list(APPEND CMAKE_MODULE_PATH ${ROTOPT_EXTERNAL}/Catch2/contrib)
    endif()

    add_executable(unit_tests unit_tests.cc)
    set_target_properties(unit_tests PROPERTIES CXX_STANDARD 14)

    target_link_libraries(unit_tests PUBLIC
        Catch2::Catch2
        rotation_optimization
        warnings::all
    )

    target_compile_options(unit_tests PRIVATE -Wno-unused-parameter)
endif()
