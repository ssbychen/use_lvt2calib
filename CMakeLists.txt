cmake_minimum_required(VERSION 3.10)
project(lvt2calib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL    REQUIRED COMPONENTS common io registration visualization)
find_package(Ceres  REQUIRED)

# yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
    set(YAML_CPP_LIBRARIES ${YAML_CPP_LIBRARIES})
    set(YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIRS})
else()
    set(YAML_CPP_LIBRARIES yaml-cpp)
    set(YAML_CPP_INCLUDE_DIRS "")
endif()

# ── Include / link dirs ───────────────────────────────────────────────────────
include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ── extrinsic_calib_l2c ───────────────────────────────────────────────────────
add_executable(extrinsic_calib_l2c
    src/ex_calib/extrinsic_calib_l2c.cpp
)
target_link_libraries(extrinsic_calib_l2c
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)

# ── extrinsic_calib_l2l ───────────────────────────────────────────────────────
add_executable(extrinsic_calib_l2l
    src/ex_calib/extrinsic_calib_l2l.cpp
)
target_link_libraries(extrinsic_calib_l2l
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
)
