cmake_minimum_required(VERSION 3.5)
project(by_your_command)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
ament_python_install_package(silero_vad)
ament_python_install_package(interactions)

# Create symlinks for console scripts in lib/${PROJECT_NAME}
install(PROGRAMS
  silero_vad/silero_vad_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME silero_vad_node
)

install(PROGRAMS
  interactions/interaction_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME interaction_node
)

install(PROGRAMS
  silero_vad/speech_only.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME speech_only
)

# Install configuration, launch files, and other resources
install(DIRECTORY 
  bringup
  config
  devrules
  DESTINATION share/${PROJECT_NAME}
)

# Install package manifest
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
