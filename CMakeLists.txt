cmake_minimum_required(VERSION 3.20)

project(ForceWM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ForceWMDependencies)

forcewm_configure_runtime_dependencies()
forcewm_configure_opensai_dependencies()
find_package(Threads REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(FORCEWM_SIMULATION_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/simulation.cpp")
file(SIZE "${FORCEWM_SIMULATION_SOURCE}" FORCEWM_SIMULATION_SOURCE_SIZE)

if(FORCEWM_SIMULATION_SOURCE_SIZE GREATER 0)
  add_executable(forcewm_simulation "${FORCEWM_SIMULATION_SOURCE}")
  target_link_libraries(forcewm_simulation PRIVATE
    forcewm::runtime
    forcewm::opensai_model
    Threads::Threads
    yaml-cpp::yaml-cpp
    ${OpenCV_LIBS}
  )

  set_target_properties(forcewm_simulation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  )
else()
  add_custom_target(forcewm_simulation SOURCES "${FORCEWM_SIMULATION_SOURCE}")
  message(STATUS "src/simulation.cpp is empty, so no executable is built yet.")
endif()
