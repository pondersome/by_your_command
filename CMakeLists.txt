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
find_package(std_msgs REQUIRED)
find_package(audio_common_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioDataUtterance.msg"
  "msg/AudioDataUtteranceStamped.msg"
  DEPENDENCIES std_msgs audio_common_msgs
)

# Install Python modules
ament_python_install_package(audio)
ament_python_install_package(interactions)
ament_python_install_package(ros_ai_bridge)
ament_python_install_package(agents)
ament_python_install_package(tests)

# Create symlinks for console scripts in lib/${PROJECT_NAME}
install(PROGRAMS
  audio/silero_vad_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME silero_vad_node
)

install(PROGRAMS
  interactions/interaction_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME interaction_node
)

install(PROGRAMS
  audio/voice_chunk_recorder.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME voice_chunk_recorder
)

install(PROGRAMS
  ros_ai_bridge/ros_ai_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME ros_ai_bridge
)

install(PROGRAMS
  agents/oai_realtime/main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME oai_realtime_agent
)

install(PROGRAMS
  agents/gemini_live/main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gemini_live_agent
)

install(PROGRAMS
  audio/simple_audio_player.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME simple_audio_player
)

install(PROGRAMS
  audio/file_audio_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME file_audio_publisher
)

install(PROGRAMS
  audio/echo_suppressor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME echo_suppressor
)

install(PROGRAMS
  audio/clap_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME clap_detector_node
)

install(PROGRAMS
  nodes/command_processor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME command_processor
)

install(PROGRAMS
  tests/test_command_processor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_command_processor
)

install(PROGRAMS
  tests/publish_command.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME publish_command
)

install(PROGRAMS
  tests/test_external_clap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_external_clap
)

install(PROGRAMS
  tests/test_external_clap_tuned.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_external_clap_tuned
)

install(PROGRAMS
  tests/test_clap_frequency_analysis.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_clap_frequency_analysis
)

install(PROGRAMS
  tests/test_simple_energy_clap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_simple_energy_clap
)

install(PROGRAMS
  tests/test_sliding_window_clap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_sliding_window_clap
)

install(PROGRAMS
  tests/test_spectral_clap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_spectral_clap
)

install(PROGRAMS
  tests/test_transient_clap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_transient_clap
)

install(PROGRAMS
  tests/audio_recorder.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME audio_recorder
)

install(PROGRAMS
  tests/test_data_driven_clap.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME test_data_driven_clap
)

install(PROGRAMS
  scripts/expand_prompt.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME expand_prompt
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

# Export dependencies
ament_export_dependencies(
  std_msgs
  audio_common_msgs
  rosidl_default_runtime
)

ament_package()
