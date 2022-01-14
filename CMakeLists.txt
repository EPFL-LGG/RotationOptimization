################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.14)
project(rotation_optimization)

# CMP0063: Honor visibility properties for all target types.
if (POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

set(CMAKE_MACOSX_RPATH 1)

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
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_definitions(-fdiagnostics-color=always)
endif()

# Export compile flags(used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-comment")

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Work-around issue where include directories of imported targets are not
# interpreted as SYSTEM contrary to the claims here:
# https://discourse.cmake.org/t/declare-imported-targets-as-system-to-suppress-warnings/555
include(target_link_libraries_system)

################################################################################
# Common libraries
################################################################################
include(FetchContent)
set(ROTOPT_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(ROTOPT_EXTERNAL "${ROTOPT_ROOT}/3rdparty")

function(rotopt_download_project name)
    FetchContent_Declare(${name}
        SOURCE_DIR   ${ROTOPT_EXTERNAL}/${name}
        DOWNLOAD_DIR ${ROTOPT_EXTERNAL}/.cache/${name}
        QUIET
        ${ARGN}
    )
    set(OLD_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE Release)
    FetchContent_MakeAvailable(${name})
    set(CMAKE_BUILD_TYPE ${OLD_CMAKE_BUILD_TYPE})
endfunction()


# Eigen3 library
if(NOT TARGET Eigen3::Eigen)
    rotopt_download_project(Eigen
        URL     https://gitlab.com/libeigen/eigen/-/archive/3.4-rc1/eigen-3.4-rc1.tar.gz
        URL_MD5 0839b9721e65d2328fb96eb4290d74cc
    )
endif()

################################################################################
# Rotation optimization library
################################################################################
add_library(rotation_optimization INTERFACE)
target_link_libraries_system(rotation_optimization INTERFACE Eigen3::Eigen)
target_include_directories(rotation_optimization INTERFACE .)

################################################################################
# Unit tests
################################################################################
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    # Catch2
    if(NOT TARGET Catch2::Catch2)
        rotopt_download_project(Catch2
            URL     https://github.com/catchorg/Catch2/archive/v2.13.3.tar.gz
            URL_MD5 57612324e1e0b49dfc8eab68c03f8473
        )
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
