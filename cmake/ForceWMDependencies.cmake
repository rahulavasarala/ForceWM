include_guard(GLOBAL)

get_filename_component(FORCEWM_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(FORCEWM_CORE_DIR "${FORCEWM_REPO_ROOT}/core")

if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  list(PREPEND CMAKE_INCLUDE_PATH
       "$ENV{CONDA_PREFIX}/include"
       "$ENV{CONDA_PREFIX}/include/eigen3")
  list(PREPEND CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
endif()

function(forcewm_configure_runtime_dependencies)
  if(TARGET forcewm::runtime)
    return()
  endif()

  find_package(glfw3 CONFIG REQUIRED)
  find_package(mujoco CONFIG QUIET)

  if(TARGET mujoco::mujoco)
    set(FORCEWM_MUJOCO_TARGET mujoco::mujoco)
  else()
    find_path(FORCEWM_MUJOCO_INCLUDE_DIR mujoco/mujoco.h REQUIRED)
    find_library(FORCEWM_MUJOCO_LIBRARY mujoco REQUIRED)

    add_library(forcewm_mujoco INTERFACE)
    add_library(forcewm::mujoco ALIAS forcewm_mujoco)
    target_include_directories(forcewm_mujoco INTERFACE
      "${FORCEWM_MUJOCO_INCLUDE_DIR}")
    target_link_libraries(forcewm_mujoco INTERFACE
      "${FORCEWM_MUJOCO_LIBRARY}")
    set(FORCEWM_MUJOCO_TARGET forcewm::mujoco)
  endif()

  add_library(forcewm_runtime INTERFACE)
  add_library(forcewm::runtime ALIAS forcewm_runtime)

  target_include_directories(forcewm_runtime INTERFACE
    "${FORCEWM_REPO_ROOT}/src"
    "${FORCEWM_REPO_ROOT}/models"
  )

  target_link_libraries(forcewm_runtime INTERFACE
    glfw
    ${FORCEWM_MUJOCO_TARGET}
  )

  target_compile_definitions(forcewm_runtime INTERFACE
    FORCEWM_MODEL_ROOT="${FORCEWM_REPO_ROOT}/models"
  )
endfunction()

function(forcewm_configure_opensai_dependencies)
  if(TARGET forcewm::opensai_model)
    return()
  endif()

  find_package(Eigen3 REQUIRED)
  find_package(jsoncpp CONFIG REQUIRED)
  forcewm_find_opensai_package("SAI-URDF" "sai-urdfreader")
  forcewm_find_opensai_package("SAI-MODEL" "sai-model")
  forcewm_find_opensai_package("SAI-COMMON" "sai-common")
  forcewm_find_opensai_package("SAI-PRIMITIVES" "sai-primitives")

  add_library(forcewm_opensai_model INTERFACE)
  add_library(forcewm::opensai_model ALIAS forcewm_opensai_model)

  target_include_directories(forcewm_opensai_model INTERFACE
    ${SAI-URDF_INCLUDE_DIRS}
    ${SAI-MODEL_INCLUDE_DIRS}
    ${SAI-COMMON_INCLUDE_DIRS}
    ${SAI-PRIMITIVES_INCLUDE_DIRS}
  )

  target_link_libraries(forcewm_opensai_model INTERFACE
    Eigen3::Eigen
    ${SAI-URDF_LIBRARIES}
    ${SAI-MODEL_LIBRARIES}
    ${SAI-COMMON_LIBRARIES}
    ${SAI-PRIMITIVES_LIBRARIES}
  )
endfunction()

function(forcewm_find_opensai_package package_name repo_dir)
  set(package_build_dir "${FORCEWM_CORE_DIR}/${repo_dir}/build")
  if(NOT EXISTS "${package_build_dir}")
    message(FATAL_ERROR
      "Missing ${package_name} at ${package_build_dir}. "
      "Run `bash installation_scripts/install_core_libraries.sh` first.")
  endif()

  set(${package_name}_DIR "${package_build_dir}" CACHE PATH
      "Build directory for ${package_name}" FORCE)
  find_package(${package_name} REQUIRED CONFIG)

  foreach(var_suffix IN ITEMS INCLUDE_DIRS LIBRARIES DEFINITIONS)
    if(DEFINED ${package_name}_${var_suffix})
      set(${package_name}_${var_suffix} "${${package_name}_${var_suffix}}" PARENT_SCOPE)
    endif()
  endforeach()
endfunction()
