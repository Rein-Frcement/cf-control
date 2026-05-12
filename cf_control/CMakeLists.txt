cmake_minimum_required(VERSION 3.8)
project(cf_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(actuator_msgs REQUIRED)
find_package(cf_control_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# libraries
add_library(mixer_lib SHARED src/mixer.cpp)
ament_target_dependencies(mixer_lib rclcpp rclcpp_components actuator_msgs cf_control_msgs nav_msgs)
target_include_directories(mixer_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
rclcpp_components_register_node(
    mixer_lib
    PLUGIN "evs::cf::Mixer"
    EXECUTABLE mixer
)

# install shared libraries
install(TARGETS
  mixer_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
ament_export_targets(export_${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
