cmake_minimum_required(VERSION 3.16)

project(RGN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(rgn_core STATIC
    src/attribute_index.cpp
    src/region_guidance.cpp)

target_include_directories(rgn_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(rgn_core PUBLIC /W4)
else()
    target_compile_options(rgn_core PUBLIC -Wall -Wextra -Wpedantic)
endif()

add_executable(rgn
    app/rgn.cpp
    app/config.cpp
    app/dataset.cpp
    app/workload.cpp
    app/experiment.cpp)

target_link_libraries(rgn PRIVATE rgn_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(rgn_tests tests/multi_attribute_test.cpp)
    target_link_libraries(rgn_tests PRIVATE rgn_core)
    add_test(NAME rgn_multi_attribute COMMAND rgn_tests)

    add_executable(rgn_end_to_end_tests
        tests/end_to_end_test.cpp
        app/dataset.cpp
        app/workload.cpp
        app/experiment.cpp)
    target_link_libraries(rgn_end_to_end_tests PRIVATE rgn_core)
    add_test(NAME rgn_end_to_end COMMAND rgn_end_to_end_tests)
endif()
