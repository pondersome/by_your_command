# OpenAI Realtime Agent Sequence Analysis

## This is a log of a Dual OpenAI Realtime Agent system startup through first response

~/ros2_ws$ ros2 launch by_your_command oai_dual_agent.launch.py namespace:=grunt1 prefix:=agent
[INFO] [launch]: All log files can be found below /home/karim/.ros/log/2025-08-20-08-34-23-462782-barney-1028443
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [launch.user]: 🚀 Starting Dual Agent System
🗣️  Conversational Agent:
    Model: gpt-4o-realtime-preview
    Voice: alloy
🤖 Command Extraction Agent:
    Model: gpt-4o-realtime-preview
    Topics: /command_transcript, /command_detected
⏱️  Pause timeout: 10.0s
🔊 Both agents listening for speech...
[INFO] [launch.user]: Command extractor will publish to /command_transcript and /command_detected
[INFO] [audio_capturer_node-1]: process started with pid [1028447]
[INFO] [simple_audio_player-2]: process started with pid [1028449]
[INFO] [silero_vad_node-3]: process started with pid [1028451]
[INFO] [ros_ai_bridge-4]: process started with pid [1028453]
[INFO] [command_processor-5]: process started with pid [1028455]
[INFO] [voice_chunk_recorder-6]: process started with pid [1028457]
[INFO] [oai_realtime_agent-7]: process started with pid [1028459]
[INFO] [oai_realtime_agent-8]: process started with pid [1028461]
[ros_ai_bridge-4] /home/karim/ros2_ws/install/by_your_command/lib/by_your_command/ros_ai_bridge:33: DeprecationWarning: websockets.WebSocketServerProtocol is deprecated
[ros_ai_bridge-4]   from websockets import WebSocketServerProtocol
[audio_capturer_node-1] ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
[audio_capturer_node-1] ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
[audio_capturer_node-1] ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
[audio_capturer_node-1] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[audio_capturer_node-1] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[audio_capturer_node-1] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[audio_capturer_node-1] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[audio_capturer_node-1] ALSA lib pcm_oss.c:397:(_snd_pcm_oss_open) Cannot open device /dev/dsp
[audio_capturer_node-1] ALSA lib pcm_oss.c:397:(_snd_pcm_oss_open) Cannot open device /dev/dsp
[audio_capturer_node-1] ALSA lib confmisc.c:160:(snd_config_get_card) Invalid field card
[audio_capturer_node-1] ALSA lib pcm_usb_stream.c:482:(_snd_pcm_usb_stream_open) Invalid card 'card'
[audio_capturer_node-1] ALSA lib confmisc.c:160:(snd_config_get_card) Invalid field card
[audio_capturer_node-1] ALSA lib pcm_usb_stream.c:482:(_snd_pcm_usb_stream_open) Invalid card 'card'
[audio_capturer_node-1] [INFO] [1755696864.497464236] [grunt1.agent.audio_capturer_node]: AudioCapturer node started
[ros_ai_bridge-4] [INFO] [1755696865.017896901] [grunt1.agent.ros_ai_bridge]: [08:34:24.925] [bridge] Processing subscribed_topics: type=<class 'list'>, value=[{'topic': 'voice_chunks', 'msg_type': 'by_your_command/AudioDataUtterance'}, {'topic': 'text_input', 'msg_type': 'std_msgs/String'}, {'topic': 'conversation_id', 'msg_type': 'std_msgs/String'}]
[ros_ai_bridge-4] [INFO] [1755696865.019734033] [grunt1.agent.ros_ai_bridge]: [08:34:25.018] [bridge] Processing published_topics: type=<class 'list'>, value=[{'topic': 'audio_out', 'msg_type': 'audio_common_msgs/AudioData'}, {'topic': 'llm_transcript', 'msg_type': 'std_msgs/String'}, {'topic': 'command_transcript', 'msg_type': 'std_msgs/String'}, {'topic': 'cmd_vel', 'msg_type': 'geometry_msgs/Twist'}, {'topic': 'conversation_id', 'msg_type': 'std_msgs/String'}, {'topic': 'interruption_signal', 'msg_type': 'std_msgs/Bool'}]
[ros_ai_bridge-4] [INFO] [1755696865.021619075] [grunt1.agent.ros_ai_bridge]: [08:34:25.019] [bridge] Loaded configuration from /home/karim/ros2_ws/install/by_your_command/share/by_your_command/config/bridge_dual_agent.yaml
[ros_ai_bridge-4] [INFO] [1755696865.024108180] [grunt1.agent.ros_ai_bridge]: [08:34:25.022] [bridge] ROS AI Bridge initialized
[ros_ai_bridge-4] [INFO] [1755696865.028002584] [grunt1.agent.ros_ai_bridge]: [08:34:25.027] [bridge] Stored asyncio loop reference: 135297461018080
[ros_ai_bridge-4] [INFO] [1755696865.029441173] [grunt1.agent.ros_ai_bridge]: Starting ROS2 node spinning...
[voice_chunk_recorder-6] [INFO] [1755696865.030297621] [grunt1.agent.voice_recorder_output]: Starting in audio_data mode, subscribing to audio_out
[ros_ai_bridge-4] [INFO] [1755696865.032009830] [grunt1.agent.ros_ai_bridge]: [08:34:25.028] [bridge] Setting up topics: subscribed_topics type=<class 'list'>, value=[{'topic': 'voice_chunks', 'msg_type': 'by_your_command/AudioDataUtterance'}, {'topic': 'text_input', 'msg_type': 'std_msgs/String'}, {'topic': 'conversation_id', 'msg_type': 'std_msgs/String'}]
[ros_ai_bridge-4] [INFO] [1755696865.034085981] [grunt1.agent.ros_ai_bridge]: [08:34:25.032] [bridge] Setting up 3 subscriptions
[voice_chunk_recorder-6] [INFO] [1755696865.035868783] [grunt1.agent.voice_recorder_output]: Input sample rate: 16000 Hz, timeout: 10.0 seconds
[ros_ai_bridge-4] [INFO] [1755696865.039348394] [grunt1.agent.ros_ai_bridge]: [08:34:25.034] [bridge] Processing subscription config: {'topic': 'voice_chunks', 'msg_type': 'by_your_command/AudioDataUtterance'}
[ros_ai_bridge-4] [INFO] [1755696865.040053506] [grunt1.agent.ros_ai_bridge]: [08:34:25.039] [bridge] Subscribing to /grunt1/agent/voice_chunks (base: voice_chunks)
[ros_ai_bridge-4] [INFO] [1755696865.078854594] [grunt1.agent.ros_ai_bridge]: [08:34:25.078] [bridge] Added subscription to /grunt1/agent/voice_chunks (by_your_command/AudioDataUtterance)
[ros_ai_bridge-4] [INFO] [1755696865.089654057] [grunt1.agent.ros_ai_bridge]: [08:34:25.078] [bridge] Processing subscription config: {'topic': 'text_input', 'msg_type': 'std_msgs/String'}
[ros_ai_bridge-4] [INFO] [1755696865.094605915] [grunt1.agent.ros_ai_bridge]: [08:34:25.089] [bridge] Subscribing to /grunt1/agent/text_input (base: text_input)
[simple_audio_player-2] ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear
[simple_audio_player-2] ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.center_lfe
[simple_audio_player-2] ALSA lib pcm.c:2664:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.side
[simple_audio_player-2] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[simple_audio_player-2] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[simple_audio_player-2] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[simple_audio_player-2] ALSA lib pcm_route.c:877:(find_matching_chmap) Found no matching channel map
[simple_audio_player-2] ALSA lib pcm_oss.c:397:(_snd_pcm_oss_open) Cannot open device /dev/dsp
[simple_audio_player-2] ALSA lib pcm_oss.c:397:(_snd_pcm_oss_open) Cannot open device /dev/dsp
[simple_audio_player-2] ALSA lib confmisc.c:160:(snd_config_get_card) Invalid field card
[simple_audio_player-2] ALSA lib pcm_usb_stream.c:482:(_snd_pcm_usb_stream_open) Invalid card 'card'
[simple_audio_player-2] ALSA lib confmisc.c:160:(snd_config_get_card) Invalid field card
[simple_audio_player-2] ALSA lib pcm_usb_stream.c:482:(_snd_pcm_usb_stream_open) Invalid card 'card'
[command_processor-5] [INFO] [1755696865.260638020] [grunt1.agent.command_processor]: Command Processor initialized
[command_processor-5] [INFO] [1755696865.264276851] [grunt1.agent.command_processor]:   Listening on: command_transcript
[command_processor-5] [INFO] [1755696865.264778126] [grunt1.agent.command_processor]:   Publishing arm presets to: /grunt1/arm_preset
[command_processor-5] [INFO] [1755696865.265659346] [grunt1.agent.command_processor]:   Publishing behavior commands to: /grunt1/behavior_command
[ros_ai_bridge-4] [INFO] [1755696865.282824892] [grunt1.agent.ros_ai_bridge]: [08:34:25.281] [bridge] Added subscription to /grunt1/agent/text_input (std_msgs/String)
[ros_ai_bridge-4] [INFO] [1755696865.283392692] [grunt1.agent.ros_ai_bridge]: [08:34:25.282] [bridge] Processing subscription config: {'topic': 'conversation_id', 'msg_type': 'std_msgs/String'}
[ros_ai_bridge-4] [INFO] [1755696865.287248506] [grunt1.agent.ros_ai_bridge]: [08:34:25.286] [bridge] Subscribing to /grunt1/agent/conversation_id (base: conversation_id)
[ros_ai_bridge-4] [INFO] [1755696865.309285588] [grunt1.agent.ros_ai_bridge]: [08:34:25.302] [bridge] Added subscription to /grunt1/agent/conversation_id (std_msgs/String)
[ros_ai_bridge-4] [INFO] [1755696865.309855805] [grunt1.agent.ros_ai_bridge]: [08:34:25.309] [bridge] Setting up 6 publishers
[ros_ai_bridge-4] [INFO] [1755696865.310739853] [grunt1.agent.ros_ai_bridge]: [08:34:25.309] [bridge] Processing publisher config: {'topic': 'audio_out', 'msg_type': 'audio_common_msgs/AudioData'}
[ros_ai_bridge-4] [INFO] [1755696865.326694850] [grunt1.agent.ros_ai_bridge]: [08:34:25.324] [bridge] Created publisher for /grunt1/agent/audio_out (base: audio_out, type: audio_common_msgs/AudioData)
[ros_ai_bridge-4] [INFO] [1755696865.327702988] [grunt1.agent.ros_ai_bridge]: [08:34:25.326] [bridge] Processing publisher config: {'topic': 'llm_transcript', 'msg_type': 'std_msgs/String'}
[ros_ai_bridge-4] [INFO] [1755696865.335662018] [grunt1.agent.ros_ai_bridge]: [08:34:25.335] [bridge] Created publisher for /grunt1/agent/llm_transcript (base: llm_transcript, type: std_msgs/String)
[ros_ai_bridge-4] [INFO] [1755696865.341330553] [grunt1.agent.ros_ai_bridge]: [08:34:25.335] [bridge] Processing publisher config: {'topic': 'command_transcript', 'msg_type': 'std_msgs/String'}
[ros_ai_bridge-4] [INFO] [1755696865.349235876] [grunt1.agent.ros_ai_bridge]: [08:34:25.348] [bridge] Created publisher for /grunt1/agent/command_transcript (base: command_transcript, type: std_msgs/String)
[ros_ai_bridge-4] [INFO] [1755696865.349719610] [grunt1.agent.ros_ai_bridge]: [08:34:25.349] [bridge] Processing publisher config: {'topic': 'cmd_vel', 'msg_type': 'geometry_msgs/Twist'}
[simple_audio_player-2] [INFO] [1755696865.359641464] [grunt1.agent.simple_audio_player]: Simple audio player started on topic audio_out (16000Hz, 1 channel(s), device 19)
[oai_realtime_agent-8] [08:34:25.362] [agent:cmd] Loaded 11 macros: ['robot_name', 'robot_capabilities', 'arm_presets', 'bearing_presets', 'motion_commands', 'compound_commands', 'cmd_response', 'visual_cmd_response_format', 'visual_convo_response_format', 'personality_traits', 'first_person_references']
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] Expanded 8 macros in prompt 'barney_command_visual'
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] Expanded 5 macros in prompt 'barney_conversational'
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] Expanded 3 macros in prompt 'barney_command_extractor'
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] Expanded 2 macros in prompt 'barney_conversational_gemini'
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] Expanded 4 macros in prompt 'barney_command_extractor_gemini'
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] Loaded 7 prompts from /home/karim/ros2_ws/src/by_your_command/config/prompts.yaml
[oai_realtime_agent-8] [08:34:25.363] [agent:cmd] 🎭 Conversation monitor initialized - ID: conv_20250820_083425_363832, timeout: 600.0s
[oai_realtime_agent-8] [08:34:25.364] [agent:cmd] Initializing OpenAI Realtime Agent...
[oai_realtime_agent-8] [08:34:25.364] [agent:cmd] 🔄 Started conversation timeout monitoring
[oai_realtime_agent-8] [08:34:25.364] [agent:cmd] 🔑 OpenAI API key configured: sk-proj-...4VkA
[oai_realtime_agent-8] [08:34:25.364] [agent:cmd] Connecting to ROS AI Bridge via WebSocket...
[oai_realtime_agent-8] [08:34:25.364] [agent:cmd] Initial connection attempt 1/10
[oai_realtime_agent-8] [08:34:25.364] [agent:cmd] Connecting to bridge at ws://localhost:8765 (attempt 1)
[simple_audio_player-2] [INFO] [1755696865.382510846] [grunt1.agent.simple_audio_player]: Audio device test successful
[oai_realtime_agent-8] [08:34:25.404] [agent:cmd] Connection failed: [Errno 111] Connect call failed ('127.0.0.1', 8765)
[oai_realtime_agent-8] [08:34:25.405] [agent:cmd] Connection failed, retrying in 5.0s...
[ros_ai_bridge-4] [INFO] [1755696865.762761825] [grunt1.agent.ros_ai_bridge]: [08:34:25.759] [bridge] Created publisher for /grunt1/agent/cmd_vel (base: cmd_vel, type: geometry_msgs/Twist)
[ros_ai_bridge-4] [INFO] [1755696865.763390018] [grunt1.agent.ros_ai_bridge]: [08:34:25.762] [bridge] Processing publisher config: {'topic': 'conversation_id', 'msg_type': 'std_msgs/String'}
[ros_ai_bridge-4] [INFO] [1755696865.781997754] [grunt1.agent.ros_ai_bridge]: [08:34:25.774] [bridge] Created publisher for /grunt1/agent/conversation_id (base: conversation_id, type: std_msgs/String)
[ros_ai_bridge-4] [INFO] [1755696865.787890176] [grunt1.agent.ros_ai_bridge]: [08:34:25.782] [bridge] Processing publisher config: {'topic': 'interruption_signal', 'msg_type': 'std_msgs/Bool'}
[ros_ai_bridge-4] [INFO] [1755696865.791560042] [grunt1.agent.ros_ai_bridge]: [08:34:25.790] [bridge] Created publisher for /grunt1/agent/interruption_signal (base: interruption_signal, type: std_msgs/Bool)
[ros_ai_bridge-4] [INFO] [1755696865.794645631] [grunt1.agent.ros_ai_bridge]: [08:34:25.794] [bridge] WebSocket server started on 0.0.0.0:8765
[ros_ai_bridge-4] [INFO] [1755696865.796949487] [grunt1.agent.ros_ai_bridge]: [08:34:25.794] [bridge] WebSocket server enabled on 0.0.0.0:8765
[ros_ai_bridge-4] [INFO] [1755696865.802243858] [grunt1.agent.ros_ai_bridge]: [08:34:25.798] [bridge] ROS AI Bridge started successfully
[oai_realtime_agent-7] [08:34:25.832] [agent:conv] Loaded 11 macros: ['robot_name', 'robot_capabilities', 'arm_presets', 'bearing_presets', 'motion_commands', 'compound_commands', 'cmd_response', 'visual_cmd_response_format', 'visual_convo_response_format', 'personality_traits', 'first_person_references']
[oai_realtime_agent-7] [08:34:25.832] [agent:conv] Expanded 8 macros in prompt 'barney_command_visual'
[oai_realtime_agent-7] [08:34:25.832] [agent:conv] Expanded 5 macros in prompt 'barney_conversational'
[oai_realtime_agent-7] [08:34:25.832] [agent:conv] Expanded 3 macros in prompt 'barney_command_extractor'
[oai_realtime_agent-7] [08:34:25.832] [agent:conv] Expanded 2 macros in prompt 'barney_conversational_gemini'
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] Expanded 4 macros in prompt 'barney_command_extractor_gemini'
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] Loaded 7 prompts from /home/karim/ros2_ws/src/by_your_command/config/prompts.yaml
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] 🎭 Conversation monitor initialized - ID: conv_20250820_083425_833244, timeout: 600.0s
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] Initializing OpenAI Realtime Agent...
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] 🔄 Started conversation timeout monitoring
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] 🔑 OpenAI API key configured: sk-proj-...4VkA
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] Connecting to ROS AI Bridge via WebSocket...
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] Initial connection attempt 1/10
[oai_realtime_agent-7] [08:34:25.833] [agent:conv] Connecting to bridge at ws://localhost:8765 (attempt 1)
[ros_ai_bridge-4] [INFO] [1755696865.854085274] [grunt1.agent.ros_ai_bridge]: [08:34:25.852] [bridge] New WebSocket connection from ('127.0.0.1', 39110)
[oai_realtime_agent-7] [08:34:25.855] [agent:conv] Agent registered successfully. Session: sess_openai_realtime_1755696865
[oai_realtime_agent-7] [08:34:25.855] [agent:conv] Bridge namespace: /grunt1/agent
[oai_realtime_agent-7] [08:34:25.855] [agent:conv] ✅ Connected to bridge at ws://localhost:8765
[oai_realtime_agent-7] [08:34:25.855] [agent:conv] ✅ Successfully connected to bridge via WebSocket
[oai_realtime_agent-7] [08:34:25.855] [agent:conv] Agent initialized successfully
[oai_realtime_agent-7] [08:34:25.855] [agent:conv] 🎭 Initial conversation ID: conv_20250820_083425_833244
[ros_ai_bridge-4] [INFO] [1755696865.855993018] [grunt1.agent.ros_ai_bridge]: [08:34:25.855] [bridge] Registered agent: openai_realtime with capabilities: ['audio_processing', 'realtime_api']
[ros_ai_bridge-4] [INFO] [1755696865.856512723] [grunt1.agent.ros_ai_bridge]: [08:34:25.856] [bridge] Agent openai_realtime subscribed to topics: ['voice_chunks', 'text_input', 'conversation_id']
[oai_realtime_agent-7] [08:34:26.857] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "conversation_id", "msg_type": "std_msgs/String", "data": {"da...
[oai_realtime_agent-7] [08:34:26.857] [agent:conv] ✅ Successfully sent message to topic: conversation_id
[oai_realtime_agent-7] [08:34:26.857] [agent:conv] 📤 Published conversation ID: conv_20250820_083425_833244
[oai_realtime_agent-7] [08:34:26.858] [agent:conv] [83425_833244] Starting OpenAI Realtime Agent main loop...
[ros_ai_bridge-4] [INFO] [1755696866.860611408] [grunt1.agent.ros_ai_bridge]: [08:34:26.859] [bridge] 📥 Received outbound message: topic=conversation_id, type=std_msgs/String
[ros_ai_bridge-4] [INFO] [1755696866.861206177] [grunt1.agent.ros_ai_bridge]: [08:34:26.860] [bridge] 📤 Queueing outbound message for topic: conversation_id
[ros_ai_bridge-4] [INFO] [1755696866.863701353] [grunt1.agent.ros_ai_bridge]: [08:34:26.861] [bridge] ✅ Successfully queued outbound message for conversation_id
[ros_ai_bridge-4] [INFO] [1755696866.864562973] [grunt1.agent.ros_ai_bridge]: [08:34:26.862] [bridge] 📤 Processing outbound message from queue for topic: conversation_id
[ros_ai_bridge-4] [INFO] [1755696866.865504055] [grunt1.agent.ros_ai_bridge]: [08:34:26.864] [bridge] ✅ Published message to ROS topic: /conversation_id
[ros_ai_bridge-4] [INFO] [1755696866.867568876] [grunt1.agent.ros_ai_bridge]: [08:34:26.866] [bridge] 📡 Broadcasting message to WebSocket agents: std_msgs/String
[ros_ai_bridge-4] [INFO] [1755696866.870152876] [grunt1.agent.ros_ai_bridge]: [08:34:26.868] [bridge] 🔊 Broadcasting std_msgs/String from /grunt1/agent/conversation_id to 1 agents
[ros_ai_bridge-4] [INFO] [1755696866.873585201] [grunt1.agent.ros_ai_bridge]: [08:34:26.870] [bridge] Agent openai_realtime subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/conversation_id (base: conversation_id)
[ros_ai_bridge-4] [INFO] [1755696866.874192332] [grunt1.agent.ros_ai_bridge]: [08:34:26.873] [bridge] 📤 Sending to agent openai_realtime: /grunt1/agent/conversation_id
[oai_realtime_agent-7] [08:34:26.875] [agent:conv] 📥 Queued message: std_msgs/String (queue size: 1)
[oai_realtime_agent-7] [08:34:26.875] [agent:conv] 📤 Retrieved message: std_msgs/String
[oai_realtime_agent-7] [08:34:26.875] [agent:conv] [83425_833244] 📨 Processing message: std_msgs/String from /grunt1/agent/conversation_id
[oai_realtime_agent-7] [08:34:26.875] [agent:conv] 🔗 Creating OpenAI session for incoming message...
[oai_realtime_agent-7] [08:34:26.875] [agent:conv] 🔌 [8392] Connecting to LLM provider...
[oai_realtime_agent-7] [08:34:26.876] [agent:conv] OpenAI WebSocket URL: wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview
[silero_vad_node-3] [INFO] [1755696867.347785019] [grunt1.agent.silero_vad_node]: [08:34:27.346] [vad] Silero VAD requires exactly 512 samples at 16kHz
[silero_vad_node-3] [INFO] [1755696867.348314212] [grunt1.agent.silero_vad_node]: [08:34:27.347] [vad] Clap detection enabled (spike ratio: 4.0x, gap: 300-800ms)
[silero_vad_node-3] [INFO] [1755696867.349492022] [grunt1.agent.silero_vad_node]: [08:34:27.349] [vad] Audio chunk #1: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.468853478] [grunt1.agent.silero_vad_node]: [08:34:27.468] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.470688669] [grunt1.agent.silero_vad_node]: [08:34:27.468] [vad] Voice activity: False
[silero_vad_node-3] [INFO] [1755696867.472297645] [grunt1.agent.silero_vad_node]: [08:34:27.471] [vad] Audio chunk #2: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.503873243] [grunt1.agent.silero_vad_node]: [08:34:27.503] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.505010832] [grunt1.agent.silero_vad_node]: [08:34:27.504] [vad] Audio chunk #3: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.507502768] [grunt1.agent.silero_vad_node]: [08:34:27.506] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.508594033] [grunt1.agent.silero_vad_node]: [08:34:27.508] [vad] Audio chunk #4: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.510401227] [grunt1.agent.silero_vad_node]: [08:34:27.509] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.511901341] [grunt1.agent.silero_vad_node]: [08:34:27.511] [vad] Audio chunk #5: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.514472328] [grunt1.agent.silero_vad_node]: [08:34:27.513] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.515791597] [grunt1.agent.silero_vad_node]: [08:34:27.515] [vad] Audio chunk #6: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.518107037] [grunt1.agent.silero_vad_node]: [08:34:27.517] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.519428658] [grunt1.agent.silero_vad_node]: [08:34:27.518] [vad] Audio chunk #7: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.521606481] [grunt1.agent.silero_vad_node]: [08:34:27.520] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.523700700] [grunt1.agent.silero_vad_node]: [08:34:27.522] [vad] Audio chunk #8: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.525798664] [grunt1.agent.silero_vad_node]: [08:34:27.525] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.526863170] [grunt1.agent.silero_vad_node]: [08:34:27.526] [vad] Audio chunk #9: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.529169364] [grunt1.agent.silero_vad_node]: [08:34:27.528] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696867.530316397] [grunt1.agent.silero_vad_node]: [08:34:27.529] [vad] Audio chunk #10: 512 samples
[silero_vad_node-3] [DEBUG] [1755696867.531928670] [grunt1.agent.silero_vad_node]: [08:34:27.531] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696867.798870256] [grunt1.agent.ros_ai_bridge]: [08:34:27.798] [bridge] Bridge metrics - Inbound: 1, Outbound: 0, Dropped: 0, Total: 2
[ros_ai_bridge-4] [INFO] [1755696867.799368906] [grunt1.agent.ros_ai_bridge]: [08:34:27.798] [bridge] WebSocket agents: 1 connected
[ros_ai_bridge-4] [INFO] [1755696868.799315002] [grunt1.agent.ros_ai_bridge]: [08:34:28.798] [bridge] Bridge metrics - Inbound: 1, Outbound: 0, Dropped: 0, Total: 2
[ros_ai_bridge-4] [INFO] [1755696868.799844147] [grunt1.agent.ros_ai_bridge]: [08:34:28.799] [bridge] WebSocket agents: 1 connected
[ros_ai_bridge-4] [INFO] [1755696869.799386629] [grunt1.agent.ros_ai_bridge]: [08:34:29.798] [bridge] Bridge metrics - Inbound: 1, Outbound: 0, Dropped: 0, Total: 2
[ros_ai_bridge-4] [INFO] [1755696869.800147178] [grunt1.agent.ros_ai_bridge]: [08:34:29.799] [bridge] WebSocket agents: 1 connected
[oai_realtime_agent-8] [08:34:30.405] [agent:cmd] Initial connection attempt 2/10
[oai_realtime_agent-8] [08:34:30.405] [agent:cmd] Connecting to bridge at ws://localhost:8765 (attempt 2)
[ros_ai_bridge-4] [INFO] [1755696870.409385400] [grunt1.agent.ros_ai_bridge]: [08:34:30.407] [bridge] New WebSocket connection from ('127.0.0.1', 37356)
[oai_realtime_agent-8] [08:34:30.410] [agent:cmd] Agent registered successfully. Session: sess_openai_command_extractor_1755696870
[oai_realtime_agent-8] [08:34:30.410] [agent:cmd] Bridge namespace: /grunt1/agent
[oai_realtime_agent-8] [08:34:30.410] [agent:cmd] ✅ Connected to bridge at ws://localhost:8765
[oai_realtime_agent-8] [08:34:30.410] [agent:cmd] ✅ Successfully connected to bridge via WebSocket
[oai_realtime_agent-8] [08:34:30.410] [agent:cmd] Agent initialized successfully
[oai_realtime_agent-8] [08:34:30.410] [agent:cmd] 🎭 Initial conversation ID: conv_20250820_083425_363832
[ros_ai_bridge-4] [INFO] [1755696870.411572162] [grunt1.agent.ros_ai_bridge]: [08:34:30.410] [bridge] Registered agent: openai_command_extractor with capabilities: ['audio_processing', 'realtime_api']
[ros_ai_bridge-4] [INFO] [1755696870.412503213] [grunt1.agent.ros_ai_bridge]: [08:34:30.411] [bridge] Agent openai_command_extractor subscribed to topics: ['voice_chunks', 'text_input', 'conversation_id']
[silero_vad_node-3] [INFO] [1755696870.417203126] [grunt1.agent.silero_vad_node]: [08:34:30.416] [vad] Audio chunk #100: 512 samples
[silero_vad_node-3] [DEBUG] [1755696870.419912774] [grunt1.agent.silero_vad_node]: [08:34:30.419] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696870.799229164] [grunt1.agent.ros_ai_bridge]: [08:34:30.798] [bridge] Bridge metrics - Inbound: 1, Outbound: 0, Dropped: 0, Total: 2
[ros_ai_bridge-4] [INFO] [1755696870.799903526] [grunt1.agent.ros_ai_bridge]: [08:34:30.799] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-8] [08:34:31.411] [agent:cmd] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "conversation_id", "msg_type": "std_msgs/String", "data": {"da...
[oai_realtime_agent-8] [08:34:31.412] [agent:cmd] ✅ Successfully sent message to topic: conversation_id
[oai_realtime_agent-8] [08:34:31.412] [agent:cmd] 📤 Published conversation ID: conv_20250820_083425_363832
[oai_realtime_agent-8] [08:34:31.412] [agent:cmd] [83425_363832] Starting OpenAI Realtime Agent main loop...
[ros_ai_bridge-4] [INFO] [1755696871.413370628] [grunt1.agent.ros_ai_bridge]: [08:34:31.412] [bridge] 📥 Received outbound message: topic=conversation_id, type=std_msgs/String
[ros_ai_bridge-4] [INFO] [1755696871.416050455] [grunt1.agent.ros_ai_bridge]: [08:34:31.413] [bridge] 📤 Queueing outbound message for topic: conversation_id
[ros_ai_bridge-4] [INFO] [1755696871.416804003] [grunt1.agent.ros_ai_bridge]: [08:34:31.416] [bridge] ✅ Successfully queued outbound message for conversation_id
[ros_ai_bridge-4] [INFO] [1755696871.417713569] [grunt1.agent.ros_ai_bridge]: [08:34:31.417] [bridge] 📤 Processing outbound message from queue for topic: conversation_id
[ros_ai_bridge-4] [INFO] [1755696871.418251679] [grunt1.agent.ros_ai_bridge]: [08:34:31.417] [bridge] ✅ Published message to ROS topic: /conversation_id
[ros_ai_bridge-4] [INFO] [1755696871.419219899] [grunt1.agent.ros_ai_bridge]: [08:34:31.418] [bridge] 📡 Broadcasting message to WebSocket agents: std_msgs/String
[ros_ai_bridge-4] [INFO] [1755696871.421546054] [grunt1.agent.ros_ai_bridge]: [08:34:31.420] [bridge] 🔊 Broadcasting std_msgs/String from /grunt1/agent/conversation_id to 2 agents
[ros_ai_bridge-4] [INFO] [1755696871.422646123] [grunt1.agent.ros_ai_bridge]: [08:34:31.421] [bridge] Agent openai_realtime subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/conversation_id (base: conversation_id)
[ros_ai_bridge-4] [INFO] [1755696871.423791050] [grunt1.agent.ros_ai_bridge]: [08:34:31.422] [bridge] 📤 Sending to agent openai_realtime: /grunt1/agent/conversation_id
[oai_realtime_agent-7] [08:34:31.424] [agent:conv] 📥 Queued message: std_msgs/String (queue size: 1)
[ros_ai_bridge-4] [INFO] [1755696871.424608491] [grunt1.agent.ros_ai_bridge]: [08:34:31.424] [bridge] Agent openai_command_extractor subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/conversation_id (base: conversation_id)
[ros_ai_bridge-4] [INFO] [1755696871.426442558] [grunt1.agent.ros_ai_bridge]: [08:34:31.424] [bridge] 📤 Sending to agent openai_command_extractor: /grunt1/agent/conversation_id
[oai_realtime_agent-8] [08:34:31.427] [agent:cmd] 📥 Queued message: std_msgs/String (queue size: 1)
[oai_realtime_agent-8] [08:34:31.427] [agent:cmd] 📤 Retrieved message: std_msgs/String
[oai_realtime_agent-8] [08:34:31.427] [agent:cmd] [83425_363832] 📨 Processing message: std_msgs/String from /grunt1/agent/conversation_id
[oai_realtime_agent-8] [08:34:31.427] [agent:cmd] 🔗 Creating OpenAI session for incoming message...
[oai_realtime_agent-8] [08:34:31.427] [agent:cmd] 🔌 [2101] Connecting to LLM provider...
[oai_realtime_agent-8] [08:34:31.427] [agent:cmd] OpenAI WebSocket URL: wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview
[ros_ai_bridge-4] [INFO] [1755696871.799291458] [grunt1.agent.ros_ai_bridge]: [08:34:31.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696871.799934706] [grunt1.agent.ros_ai_bridge]: [08:34:31.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696872.799232761] [grunt1.agent.ros_ai_bridge]: [08:34:32.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696872.799981426] [grunt1.agent.ros_ai_bridge]: [08:34:32.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696873.587185890] [grunt1.agent.silero_vad_node]: [08:34:33.586] [vad] Audio chunk #200: 512 samples
[silero_vad_node-3] [DEBUG] [1755696873.589073571] [grunt1.agent.silero_vad_node]: [08:34:33.588] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696873.798866258] [grunt1.agent.ros_ai_bridge]: [08:34:33.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696873.799347383] [grunt1.agent.ros_ai_bridge]: [08:34:33.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696874.798976960] [grunt1.agent.ros_ai_bridge]: [08:34:34.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696874.799487889] [grunt1.agent.ros_ai_bridge]: [08:34:34.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696875.799079964] [grunt1.agent.ros_ai_bridge]: [08:34:35.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696875.799770514] [grunt1.agent.ros_ai_bridge]: [08:34:35.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696876.798877638] [grunt1.agent.ros_ai_bridge]: [08:34:36.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696876.799349172] [grunt1.agent.ros_ai_bridge]: [08:34:36.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696876.809940225] [grunt1.agent.silero_vad_node]: [08:34:36.809] [vad] Audio chunk #300: 512 samples
[silero_vad_node-3] [DEBUG] [1755696876.812634144] [grunt1.agent.silero_vad_node]: [08:34:36.811] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696877.413467583] [grunt1.agent.silero_vad_node]: [08:34:37.412] [vad] Voice activity: True
[silero_vad_node-3] [INFO] [1755696877.413973772] [grunt1.agent.silero_vad_node]: [08:34:37.413] [vad] Voice detected. Starting utterance 1755696877370204416.
[silero_vad_node-3] [INFO] [1755696877.414656863] [grunt1.agent.silero_vad_node]: [08:34:37.414] [vad] Initialized chunking buffer with 15 pre-roll frames
[ros_ai_bridge-4] [INFO] [1755696877.798863663] [grunt1.agent.ros_ai_bridge]: [08:34:37.798] [bridge] Bridge metrics - Inbound: 2, Outbound: 0, Dropped: 0, Total: 4
[ros_ai_bridge-4] [INFO] [1755696877.799324721] [grunt1.agent.ros_ai_bridge]: [08:34:37.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696878.497987561] [grunt1.agent.silero_vad_node]: [08:34:38.497] [vad] Voice activity: False
[silero_vad_node-3] [INFO] [1755696878.498472093] [grunt1.agent.silero_vad_node]: [08:34:38.498] [vad] Voice ended for utterance 1755696877370204416. Preparing final chunk.
[silero_vad_node-3] [INFO] [1755696878.498950787] [grunt1.agent.silero_vad_node]: [08:34:38.498] [vad] Publishing final chunk with 50 remaining frames
[silero_vad_node-3] [INFO] [1755696878.541834761] [grunt1.agent.silero_vad_node]: [08:34:38.539] [vad] Published end-of-utterance chunk for utterance 1755696877370204416
[ros_ai_bridge-4] [INFO] [1755696878.541999320] [grunt1.agent.ros_ai_bridge]: [08:34:38.541] [bridge] 📡 Broadcasting message to WebSocket agents: by_your_command/AudioDataUtterance
[ros_ai_bridge-4] [INFO] [1755696878.542733730] [grunt1.agent.ros_ai_bridge]: [08:34:38.542] [bridge] 🔊 Broadcasting by_your_command/AudioDataUtterance from /grunt1/agent/voice_chunks to 2 agents
[ros_ai_bridge-4] [INFO] [1755696878.548977182] [grunt1.agent.ros_ai_bridge]: [08:34:38.542] [bridge] Agent openai_realtime subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/voice_chunks (base: voice_chunks)
[ros_ai_bridge-4] [INFO] [1755696878.550628537] [grunt1.agent.ros_ai_bridge]: [08:34:38.549] [bridge] 📤 Sending to agent openai_realtime: /grunt1/agent/voice_chunks
[ros_ai_bridge-4] [INFO] [1755696878.551676556] [grunt1.agent.ros_ai_bridge]: [08:34:38.550] [bridge] 🎧 Bridge serializing int16_data: type=<class 'array.array'>, length=25600
[ros_ai_bridge-4] [INFO] [1755696878.565764142] [grunt1.agent.ros_ai_bridge]: [08:34:38.564] [bridge] Agent openai_command_extractor subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/voice_chunks (base: voice_chunks)
[ros_ai_bridge-4] [INFO] [1755696878.566926822] [grunt1.agent.ros_ai_bridge]: [08:34:38.565] [bridge] 📤 Sending to agent openai_command_extractor: /grunt1/agent/voice_chunks
[oai_realtime_agent-7] [08:34:38.567] [agent:conv] 📥 Queued message: by_your_command/AudioDataUtterance (queue size: 2)
[ros_ai_bridge-4] [INFO] [1755696878.568374685] [grunt1.agent.ros_ai_bridge]: [08:34:38.567] [bridge] 🎧 Bridge serializing int16_data: type=<class 'array.array'>, length=25600
[oai_realtime_agent-8] [08:34:38.587] [agent:cmd] 📥 Queued message: by_your_command/AudioDataUtterance (queue size: 1)
[ros_ai_bridge-4] [INFO] [1755696878.798876496] [grunt1.agent.ros_ai_bridge]: [08:34:38.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696878.799335831] [grunt1.agent.ros_ai_bridge]: [08:34:38.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696879.799222238] [grunt1.agent.ros_ai_bridge]: [08:34:39.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696879.799875615] [grunt1.agent.ros_ai_bridge]: [08:34:39.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696879.986694865] [grunt1.agent.silero_vad_node]: [08:34:39.986] [vad] Audio chunk #400: 512 samples
[silero_vad_node-3] [DEBUG] [1755696879.988394086] [grunt1.agent.silero_vad_node]: [08:34:39.987] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696880.799397468] [grunt1.agent.ros_ai_bridge]: [08:34:40.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696880.800154537] [grunt1.agent.ros_ai_bridge]: [08:34:40.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696881.799396749] [grunt1.agent.ros_ai_bridge]: [08:34:41.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696881.800098848] [grunt1.agent.ros_ai_bridge]: [08:34:41.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696882.798849857] [grunt1.agent.ros_ai_bridge]: [08:34:42.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696882.799310738] [grunt1.agent.ros_ai_bridge]: [08:34:42.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696883.196622560] [grunt1.agent.silero_vad_node]: [08:34:43.196] [vad] Audio chunk #500: 512 samples
[silero_vad_node-3] [DEBUG] [1755696883.200208471] [grunt1.agent.silero_vad_node]: [08:34:43.198] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696883.799181918] [grunt1.agent.ros_ai_bridge]: [08:34:43.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696883.799919215] [grunt1.agent.ros_ai_bridge]: [08:34:43.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696884.798864079] [grunt1.agent.ros_ai_bridge]: [08:34:44.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696884.799314144] [grunt1.agent.ros_ai_bridge]: [08:34:44.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696885.798912773] [grunt1.agent.ros_ai_bridge]: [08:34:45.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696885.799453719] [grunt1.agent.ros_ai_bridge]: [08:34:45.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696886.407231905] [grunt1.agent.silero_vad_node]: [08:34:46.406] [vad] Audio chunk #600: 512 samples
[silero_vad_node-3] [DEBUG] [1755696886.410759748] [grunt1.agent.silero_vad_node]: [08:34:46.410] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696886.798891422] [grunt1.agent.ros_ai_bridge]: [08:34:46.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696886.799377381] [grunt1.agent.ros_ai_bridge]: [08:34:46.798] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-7] [08:34:46.897] [agent:conv] ❌ [8392] Connection failed: [Errno -3] Temporary failure in name resolution
[oai_realtime_agent-7] [08:34:46.897] [agent:conv] ❌ Failed to create session - check configuration and API status
[oai_realtime_agent-7] [08:34:46.908] [agent:conv] 📤 Retrieved message: std_msgs/String
[oai_realtime_agent-7] [08:34:46.908] [agent:conv] [83425_833244] 📨 Processing message: std_msgs/String from /grunt1/agent/conversation_id
[oai_realtime_agent-7] [08:34:46.908] [agent:conv] 🔗 Creating OpenAI session for incoming message...
[oai_realtime_agent-7] [08:34:46.908] [agent:conv] 🔌 [2703] Connecting to LLM provider...
[oai_realtime_agent-7] [08:34:46.908] [agent:conv] OpenAI WebSocket URL: wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview
[ros_ai_bridge-4] [INFO] [1755696887.798921305] [grunt1.agent.ros_ai_bridge]: [08:34:47.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696887.799387880] [grunt1.agent.ros_ai_bridge]: [08:34:47.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696888.502213963] [grunt1.agent.silero_vad_node]: [08:34:48.501] [vad] Voice activity: False
[ros_ai_bridge-4] [INFO] [1755696888.798996685] [grunt1.agent.ros_ai_bridge]: [08:34:48.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696888.799432585] [grunt1.agent.ros_ai_bridge]: [08:34:48.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696889.586730549] [grunt1.agent.silero_vad_node]: [08:34:49.586] [vad] Audio chunk #700: 512 samples
[silero_vad_node-3] [DEBUG] [1755696889.588875491] [grunt1.agent.silero_vad_node]: [08:34:49.588] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696889.799193681] [grunt1.agent.ros_ai_bridge]: [08:34:49.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696889.799840159] [grunt1.agent.ros_ai_bridge]: [08:34:49.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696890.798950692] [grunt1.agent.ros_ai_bridge]: [08:34:50.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696890.799461601] [grunt1.agent.ros_ai_bridge]: [08:34:50.799] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-8] [08:34:51.448] [agent:cmd] ❌ [2101] Connection failed: [Errno -3] Temporary failure in name resolution
[oai_realtime_agent-8] [08:34:51.448] [agent:cmd] ❌ Failed to create session - check configuration and API status
[oai_realtime_agent-8] [08:34:51.459] [agent:cmd] 📤 Retrieved message: by_your_command/AudioDataUtterance
[oai_realtime_agent-8] [08:34:51.459] [agent:cmd] 🔍 WebSocket envelope int16_data: type=<class 'list'>, length=25600
[oai_realtime_agent-8] [08:34:51.459] [agent:cmd] [83425_363832] 📨 Processing message: by_your_command/AudioDataUtterance from /grunt1/agent/voice_chunks
[oai_realtime_agent-8] [08:34:51.459] [agent:cmd] 🔗 Creating OpenAI session for incoming message...
[oai_realtime_agent-8] [08:34:51.459] [agent:cmd] 🔌 [2084] Connecting to LLM provider...
[oai_realtime_agent-8] [08:34:51.459] [agent:cmd] OpenAI WebSocket URL: wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview
[ros_ai_bridge-4] [INFO] [1755696891.799187957] [grunt1.agent.ros_ai_bridge]: [08:34:51.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696891.799898564] [grunt1.agent.ros_ai_bridge]: [08:34:51.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696892.791198061] [grunt1.agent.silero_vad_node]: [08:34:52.790] [vad] Audio chunk #800: 512 samples
[silero_vad_node-3] [DEBUG] [1755696892.792964062] [grunt1.agent.silero_vad_node]: [08:34:52.792] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696892.799168742] [grunt1.agent.ros_ai_bridge]: [08:34:52.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696892.799865168] [grunt1.agent.ros_ai_bridge]: [08:34:52.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696893.799427925] [grunt1.agent.ros_ai_bridge]: [08:34:53.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696893.801180451] [grunt1.agent.ros_ai_bridge]: [08:34:53.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696894.799356373] [grunt1.agent.ros_ai_bridge]: [08:34:54.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696894.800118647] [grunt1.agent.ros_ai_bridge]: [08:34:54.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696895.798902956] [grunt1.agent.ros_ai_bridge]: [08:34:55.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696895.799354532] [grunt1.agent.ros_ai_bridge]: [08:34:55.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696896.008505747] [grunt1.agent.silero_vad_node]: [08:34:56.007] [vad] Audio chunk #900: 512 samples
[silero_vad_node-3] [DEBUG] [1755696896.010308120] [grunt1.agent.silero_vad_node]: [08:34:56.009] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696896.798895932] [grunt1.agent.ros_ai_bridge]: [08:34:56.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696896.799352479] [grunt1.agent.ros_ai_bridge]: [08:34:56.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696897.798917348] [grunt1.agent.ros_ai_bridge]: [08:34:57.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696897.799394219] [grunt1.agent.ros_ai_bridge]: [08:34:57.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696898.258281760] [grunt1.agent.silero_vad_node]: [08:34:58.257] [vad] Voice activity: True
[silero_vad_node-3] [INFO] [1755696898.258796186] [grunt1.agent.silero_vad_node]: [08:34:58.258] [vad] Voice detected. Starting utterance 1755696898215550464.
[silero_vad_node-3] [INFO] [1755696898.259296330] [grunt1.agent.silero_vad_node]: [08:34:58.258] [vad] Initialized chunking buffer with 15 pre-roll frames
[ros_ai_bridge-4] [INFO] [1755696898.799007935] [grunt1.agent.ros_ai_bridge]: [08:34:58.798] [bridge] Bridge metrics - Inbound: 3, Outbound: 0, Dropped: 0, Total: 5
[ros_ai_bridge-4] [INFO] [1755696898.799562576] [grunt1.agent.ros_ai_bridge]: [08:34:58.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696899.187663479] [grunt1.agent.silero_vad_node]: [08:34:59.187] [vad] Audio chunk #1000: 512 samples
[silero_vad_node-3] [DEBUG] [1755696899.189520249] [grunt1.agent.silero_vad_node]: [08:34:59.188] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696899.347924511] [grunt1.agent.silero_vad_node]: [08:34:59.347] [vad] Voice activity: False
[silero_vad_node-3] [INFO] [1755696899.348547871] [grunt1.agent.silero_vad_node]: [08:34:59.348] [vad] Voice ended for utterance 1755696898215550464. Preparing final chunk.
[silero_vad_node-3] [INFO] [1755696899.349710449] [grunt1.agent.silero_vad_node]: [08:34:59.348] [vad] Publishing final chunk with 51 remaining frames
[silero_vad_node-3] [INFO] [1755696899.387649509] [grunt1.agent.silero_vad_node]: [08:34:59.386] [vad] Published end-of-utterance chunk for utterance 1755696898215550464
[ros_ai_bridge-4] [INFO] [1755696899.389586468] [grunt1.agent.ros_ai_bridge]: [08:34:59.388] [bridge] 📡 Broadcasting message to WebSocket agents: by_your_command/AudioDataUtterance
[ros_ai_bridge-4] [INFO] [1755696899.393346998] [grunt1.agent.ros_ai_bridge]: [08:34:59.392] [bridge] 🔊 Broadcasting by_your_command/AudioDataUtterance from /grunt1/agent/voice_chunks to 2 agents
[ros_ai_bridge-4] [INFO] [1755696899.395663359] [grunt1.agent.ros_ai_bridge]: [08:34:59.393] [bridge] Agent openai_realtime subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/voice_chunks (base: voice_chunks)
[ros_ai_bridge-4] [INFO] [1755696899.396966966] [grunt1.agent.ros_ai_bridge]: [08:34:59.395] [bridge] 📤 Sending to agent openai_realtime: /grunt1/agent/voice_chunks
[ros_ai_bridge-4] [INFO] [1755696899.398232731] [grunt1.agent.ros_ai_bridge]: [08:34:59.397] [bridge] 🎧 Bridge serializing int16_data: type=<class 'array.array'>, length=26112
[ros_ai_bridge-4] [INFO] [1755696899.412570007] [grunt1.agent.ros_ai_bridge]: [08:34:59.410] [bridge] Agent openai_command_extractor subscriptions: ['voice_chunks', 'text_input', 'conversation_id'], checking /grunt1/agent/voice_chunks (base: voice_chunks)
[ros_ai_bridge-4] [INFO] [1755696899.413673977] [grunt1.agent.ros_ai_bridge]: [08:34:59.412] [bridge] 📤 Sending to agent openai_command_extractor: /grunt1/agent/voice_chunks
[oai_realtime_agent-7] [08:34:59.415] [agent:conv] 📥 Queued message: by_your_command/AudioDataUtterance (queue size: 2)
[ros_ai_bridge-4] [INFO] [1755696899.418491552] [grunt1.agent.ros_ai_bridge]: [08:34:59.414] [bridge] 🎧 Bridge serializing int16_data: type=<class 'array.array'>, length=26112
[oai_realtime_agent-8] [08:34:59.436] [agent:cmd] 📥 Queued message: by_your_command/AudioDataUtterance (queue size: 1)
[ros_ai_bridge-4] [INFO] [1755696899.799663403] [grunt1.agent.ros_ai_bridge]: [08:34:59.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696899.800243185] [grunt1.agent.ros_ai_bridge]: [08:34:59.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696900.799213164] [grunt1.agent.ros_ai_bridge]: [08:35:00.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696900.800764310] [grunt1.agent.ros_ai_bridge]: [08:35:00.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696901.799218919] [grunt1.agent.ros_ai_bridge]: [08:35:01.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696901.799870496] [grunt1.agent.ros_ai_bridge]: [08:35:01.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696902.408941971] [grunt1.agent.silero_vad_node]: [08:35:02.408] [vad] Audio chunk #1100: 512 samples
[silero_vad_node-3] [DEBUG] [1755696902.410795296] [grunt1.agent.silero_vad_node]: [08:35:02.410] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696902.798988132] [grunt1.agent.ros_ai_bridge]: [08:35:02.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696902.799525705] [grunt1.agent.ros_ai_bridge]: [08:35:02.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696903.799410972] [grunt1.agent.ros_ai_bridge]: [08:35:03.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696903.799900492] [grunt1.agent.ros_ai_bridge]: [08:35:03.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696904.798885676] [grunt1.agent.ros_ai_bridge]: [08:35:04.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696904.799416989] [grunt1.agent.ros_ai_bridge]: [08:35:04.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696905.587581988] [grunt1.agent.silero_vad_node]: [08:35:05.587] [vad] Audio chunk #1200: 512 samples
[silero_vad_node-3] [DEBUG] [1755696905.589375517] [grunt1.agent.silero_vad_node]: [08:35:05.588] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696905.798887064] [grunt1.agent.ros_ai_bridge]: [08:35:05.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696905.799364345] [grunt1.agent.ros_ai_bridge]: [08:35:05.798] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-8] [08:35:05.816] [agent:cmd] ❌ [2084] Connection failed: [Errno -3] Temporary failure in name resolution
[oai_realtime_agent-8] [08:35:05.816] [agent:cmd] ❌ Failed to create session - check configuration and API status
[oai_realtime_agent-7] [08:35:05.816] [agent:conv] ❌ [2703] Connection failed: [Errno -3] Temporary failure in name resolution
[oai_realtime_agent-7] [08:35:05.816] [agent:conv] ❌ Failed to create session - check configuration and API status
[oai_realtime_agent-7] [08:35:05.827] [agent:conv] 📤 Retrieved message: by_your_command/AudioDataUtterance
[oai_realtime_agent-7] [08:35:05.827] [agent:conv] 🔍 WebSocket envelope int16_data: type=<class 'list'>, length=25600
[oai_realtime_agent-7] [08:35:05.827] [agent:conv] [83425_833244] 📨 Processing message: by_your_command/AudioDataUtterance from /grunt1/agent/voice_chunks
[oai_realtime_agent-7] [08:35:05.827] [agent:conv] 🔗 Creating OpenAI session for incoming message...
[oai_realtime_agent-7] [08:35:05.827] [agent:conv] 🔌 [5060] Connecting to LLM provider...
[oai_realtime_agent-7] [08:35:05.827] [agent:conv] OpenAI WebSocket URL: wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview
[oai_realtime_agent-8] [08:35:05.829] [agent:cmd] 📤 Retrieved message: by_your_command/AudioDataUtterance
[oai_realtime_agent-8] [08:35:05.829] [agent:cmd] 🔍 WebSocket envelope int16_data: type=<class 'list'>, length=26112
[oai_realtime_agent-8] [08:35:05.829] [agent:cmd] [83425_363832] 📨 Processing message: by_your_command/AudioDataUtterance from /grunt1/agent/voice_chunks
[oai_realtime_agent-8] [08:35:05.829] [agent:cmd] 🔗 Creating OpenAI session for incoming message...
[oai_realtime_agent-8] [08:35:05.829] [agent:cmd] 🔌 [7766] Connecting to LLM provider...
[oai_realtime_agent-8] [08:35:05.829] [agent:cmd] OpenAI WebSocket URL: wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview
[ros_ai_bridge-4] [INFO] [1755696906.799238199] [grunt1.agent.ros_ai_bridge]: [08:35:06.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696906.799865706] [grunt1.agent.ros_ai_bridge]: [08:35:06.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696907.799277601] [grunt1.agent.ros_ai_bridge]: [08:35:07.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696907.799978927] [grunt1.agent.ros_ai_bridge]: [08:35:07.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696908.798872540] [grunt1.agent.ros_ai_bridge]: [08:35:08.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696908.799333901] [grunt1.agent.ros_ai_bridge]: [08:35:08.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696908.811369817] [grunt1.agent.silero_vad_node]: [08:35:08.810] [vad] Audio chunk #1300: 512 samples
[silero_vad_node-3] [DEBUG] [1755696908.813902837] [grunt1.agent.silero_vad_node]: [08:35:08.813] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[silero_vad_node-3] [INFO] [1755696909.373938724] [grunt1.agent.silero_vad_node]: [08:35:09.373] [vad] Voice activity: False
[ros_ai_bridge-4] [INFO] [1755696909.798819369] [grunt1.agent.ros_ai_bridge]: [08:35:09.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696909.799286166] [grunt1.agent.ros_ai_bridge]: [08:35:09.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696910.798878134] [grunt1.agent.ros_ai_bridge]: [08:35:10.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696910.799335621] [grunt1.agent.ros_ai_bridge]: [08:35:10.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696911.798826618] [grunt1.agent.ros_ai_bridge]: [08:35:11.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696911.799353405] [grunt1.agent.ros_ai_bridge]: [08:35:11.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696911.984316867] [grunt1.agent.silero_vad_node]: [08:35:11.983] [vad] Audio chunk #1400: 512 samples
[silero_vad_node-3] [DEBUG] [1755696911.986335343] [grunt1.agent.silero_vad_node]: [08:35:11.985] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696912.799131791] [grunt1.agent.ros_ai_bridge]: [08:35:12.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696912.799735110] [grunt1.agent.ros_ai_bridge]: [08:35:12.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696913.799184794] [grunt1.agent.ros_ai_bridge]: [08:35:13.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696913.800652054] [grunt1.agent.ros_ai_bridge]: [08:35:13.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696914.798860248] [grunt1.agent.ros_ai_bridge]: [08:35:14.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696914.799324060] [grunt1.agent.ros_ai_bridge]: [08:35:14.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696915.205921980] [grunt1.agent.silero_vad_node]: [08:35:15.205] [vad] Audio chunk #1500: 512 samples
[silero_vad_node-3] [DEBUG] [1755696915.208926840] [grunt1.agent.silero_vad_node]: [08:35:15.207] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696915.799219392] [grunt1.agent.ros_ai_bridge]: [08:35:15.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696915.799718271] [grunt1.agent.ros_ai_bridge]: [08:35:15.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696916.798872714] [grunt1.agent.ros_ai_bridge]: [08:35:16.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696916.799318192] [grunt1.agent.ros_ai_bridge]: [08:35:16.798] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696917.798862278] [grunt1.agent.ros_ai_bridge]: [08:35:17.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696917.799326758] [grunt1.agent.ros_ai_bridge]: [08:35:17.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696918.409756121] [grunt1.agent.silero_vad_node]: [08:35:18.409] [vad] Audio chunk #1600: 512 samples
[silero_vad_node-3] [DEBUG] [1755696918.411682029] [grunt1.agent.silero_vad_node]: [08:35:18.411] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696918.799356517] [grunt1.agent.ros_ai_bridge]: [08:35:18.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696918.800294211] [grunt1.agent.ros_ai_bridge]: [08:35:18.799] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-7] [08:35:18.911] [agent:conv] ✅ [5060] WebSocket connected, configuring session...
[oai_realtime_agent-7] [08:35:18.911] [agent:conv] Using voice: echo (from config)
[oai_realtime_agent-7] [08:35:18.912] [agent:conv] 📤 OpenAI session configuration sent
[oai_realtime_agent-8] [08:35:18.913] [agent:cmd] ✅ [7766] WebSocket connected, configuring session...
[oai_realtime_agent-8] [08:35:18.913] [agent:cmd] Using voice: alloy (from config)
[oai_realtime_agent-8] [08:35:18.913] [agent:cmd] 📤 OpenAI session configuration sent
[oai_realtime_agent-8] [08:35:18.913] [agent:cmd] ✅ OpenAI session created: sess_C6dGATAm9lACoExrrwubq
[oai_realtime_agent-8] [08:35:18.914] [agent:cmd] ✅ Session #1 active
[oai_realtime_agent-8] [08:35:18.914] [agent:cmd] 🚀 Starting continuous response processor task immediately
[oai_realtime_agent-8] [08:35:18.914] [agent:cmd] ✅ Session created for incoming message
[oai_realtime_agent-7] [08:35:18.914] [agent:conv] ✅ OpenAI session created: sess_C6dGAVMgrhh42UAl4FqMQ
[oai_realtime_agent-7] [08:35:18.914] [agent:conv] ✅ Session #1 active
[oai_realtime_agent-7] [08:35:18.914] [agent:conv] 🚀 Starting continuous response processor task immediately
[oai_realtime_agent-7] [08:35:18.914] [agent:conv] ✅ Session created for incoming message
[oai_realtime_agent-7] [08:35:18.925] [agent:conv] ✅ SENT chunk #0 (68318 bytes)
[oai_realtime_agent-7] [08:35:18.926] [agent:conv] 🎧 Starting continuous OpenAI response listener
[oai_realtime_agent-8] [08:35:18.930] [agent:cmd] ✅ SENT chunk #0 (69682 bytes)
[oai_realtime_agent-8] [08:35:18.933] [agent:cmd] 🎧 Starting continuous OpenAI response listener
[oai_realtime_agent-7] [08:35:19.028] [agent:conv] 💾 Committed audio buffer for utterance 1755696877370204416
[oai_realtime_agent-7] [08:35:19.028] [agent:conv] 📊 Session state: active
[oai_realtime_agent-7] [08:35:19.028] [agent:conv] ⏳ Expecting transcription + assistant response
[oai_realtime_agent-8] [08:35:19.034] [agent:cmd] 💾 Committed audio buffer for utterance 1755696898215550464
[oai_realtime_agent-8] [08:35:19.034] [agent:cmd] 📊 Session state: active
[oai_realtime_agent-8] [08:35:19.034] [agent:cmd] ⏳ Expecting transcription + assistant response
[oai_realtime_agent-7] [08:35:19.039] [agent:conv] 📤 Retrieved message: by_your_command/AudioDataUtterance
[oai_realtime_agent-7] [08:35:19.039] [agent:conv] 🔍 WebSocket envelope int16_data: type=<class 'list'>, length=26112
[oai_realtime_agent-7] [08:35:19.039] [agent:conv] [83425_833244] 📨 Processing message: by_your_command/AudioDataUtterance from /grunt1/agent/voice_chunks
[oai_realtime_agent-7] [08:35:19.052] [agent:conv] ✅ SENT chunk #0 (69682 bytes)
[oai_realtime_agent-7] [08:35:19.152] [agent:conv] 💾 Committed audio buffer for utterance 1755696898215550464
[oai_realtime_agent-7] [08:35:19.152] [agent:conv] 📊 Session state: active
[oai_realtime_agent-7] [08:35:19.152] [agent:conv] ⏳ Expecting transcription + assistant response
[silero_vad_node-3] [INFO] [1755696919.375512699] [grunt1.agent.silero_vad_node]: [08:35:19.374] [vad] Voice activity: False
[ros_ai_bridge-4] [INFO] [1755696919.799119130] [grunt1.agent.ros_ai_bridge]: [08:35:19.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696919.799700684] [grunt1.agent.ros_ai_bridge]: [08:35:19.799] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-7] [08:35:20.477] [agent:conv] 🎯 OpenAI: session.updated
[oai_realtime_agent-7] [08:35:20.477] [agent:conv] 📝 OpenAI session updated - turn_detection: server_vad
[oai_realtime_agent-8] [08:35:20.477] [agent:cmd] 🎯 OpenAI: session.updated
[oai_realtime_agent-8] [08:35:20.478] [agent:cmd] 📝 OpenAI session updated - turn_detection: server_vad
[oai_realtime_agent-7] [08:35:20.515] [agent:conv] 🎯 OpenAI: input_audio_buffer.speech_started
[oai_realtime_agent-7] [08:35:20.515] [agent:conv] 🎤 OpenAI detected speech start
[oai_realtime_agent-7] [08:35:20.575] [agent:conv] 🎯 OpenAI: input_audio_buffer.committed
[oai_realtime_agent-7] [08:35:20.575] [agent:conv] 💾 OpenAI committed audio buffer
[oai_realtime_agent-7] [08:35:20.575] [agent:conv] 🎯 OpenAI: conversation.item.created
[oai_realtime_agent-7] [08:35:20.581] [agent:conv] 🎯 OpenAI: input_audio_buffer.speech_stopped
[oai_realtime_agent-7] [08:35:20.581] [agent:conv] 🎯 OpenAI: input_audio_buffer.committed
[oai_realtime_agent-7] [08:35:20.581] [agent:conv] 💾 OpenAI committed audio buffer
[oai_realtime_agent-7] [08:35:20.582] [agent:conv] 🎯 OpenAI: conversation.item.created
[oai_realtime_agent-7] [08:35:20.582] [agent:conv] 🎯 OpenAI: conversation.item.input_audio_transcription.delta
[oai_realtime_agent-7] [08:35:20.582] [agent:conv] 🎯 OpenAI: conversation.item.input_audio_transcription.completed
[oai_realtime_agent-7] [08:35:20.582] [agent:conv] ⚠️ Empty user transcript received
[oai_realtime_agent-7] [08:35:20.582] [agent:conv] ✅ transcription complete
[oai_realtime_agent-7] [08:35:20.596] [agent:conv] 🎯 OpenAI: input_audio_buffer.speech_started
[oai_realtime_agent-7] [08:35:20.596] [agent:conv] 🎤 OpenAI detected speech start
[oai_realtime_agent-7] [08:35:20.646] [agent:conv] 🎯 OpenAI: input_audio_buffer.committed
[oai_realtime_agent-7] [08:35:20.646] [agent:conv] 💾 OpenAI committed audio buffer
[oai_realtime_agent-7] [08:35:20.646] [agent:conv] 🎯 OpenAI: conversation.item.created
[oai_realtime_agent-8] [08:35:20.671] [agent:cmd] 🎯 OpenAI: input_audio_buffer.speech_started
[oai_realtime_agent-8] [08:35:20.672] [agent:cmd] 🎤 OpenAI detected speech start
[oai_realtime_agent-8] [08:35:20.795] [agent:cmd] 🎯 OpenAI: input_audio_buffer.committed
[oai_realtime_agent-8] [08:35:20.795] [agent:cmd] 💾 OpenAI committed audio buffer
[oai_realtime_agent-8] [08:35:20.795] [agent:cmd] 🎯 OpenAI: conversation.item.created
[ros_ai_bridge-4] [INFO] [1755696920.798848640] [grunt1.agent.ros_ai_bridge]: [08:35:20.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696920.799303341] [grunt1.agent.ros_ai_bridge]: [08:35:20.798] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696921.588533865] [grunt1.agent.silero_vad_node]: [08:35:21.588] [vad] Audio chunk #1700: 512 samples
[silero_vad_node-3] [DEBUG] [1755696921.590258953] [grunt1.agent.silero_vad_node]: [08:35:21.589] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[ros_ai_bridge-4] [INFO] [1755696921.799337156] [grunt1.agent.ros_ai_bridge]: [08:35:21.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 6
[ros_ai_bridge-4] [INFO] [1755696921.799974437] [grunt1.agent.ros_ai_bridge]: [08:35:21.799] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-7] [08:35:22.252] [agent:conv] 🎯 OpenAI: conversation.item.input_audio_transcription.delta
[oai_realtime_agent-7] [08:35:22.252] [agent:conv] 🎯 OpenAI: conversation.item.input_audio_transcription.completed
[oai_realtime_agent-7] [08:35:22.252] [agent:conv] 👤 User transcript: Good morning.
[oai_realtime_agent-7] [08:35:22.252] [agent:conv] ✅ transcription complete
[oai_realtime_agent-7] [08:35:22.252] [agent:conv] 🤖 Triggering OpenAI response generation
[oai_realtime_agent-7] [08:35:22.253] [agent:conv] ✅ Response generation triggered
[oai_realtime_agent-7] [08:35:22.298] [agent:conv] 🎯 OpenAI: response.created
[oai_realtime_agent-7] [08:35:22.299] [agent:conv] 🤖 OpenAI creating response...
[oai_realtime_agent-7] [08:35:22.299] [agent:conv] ✅ assistant_response complete
[oai_realtime_agent-7] [08:35:22.562] [agent:conv] 🎯 OpenAI: response.output_item.added
[oai_realtime_agent-7] [08:35:22.562] [agent:conv] 🎯 OpenAI: conversation.item.created
[oai_realtime_agent-7] [08:35:22.562] [agent:conv] 🤖 Assistant starting response (item: item_C6dGE6b2gwVnQVvqhuGke)
[oai_realtime_agent-7] [08:35:22.591] [agent:conv] 🎯 OpenAI: response.content_part.added
[oai_realtime_agent-7] [08:35:22.714] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:22.715] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.717703692] [grunt1.agent.ros_ai_bridge]: [08:35:22.716] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696922.719299083] [grunt1.agent.ros_ai_bridge]: [08:35:22.718] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.719722638] [grunt1.agent.ros_ai_bridge]: [08:35:22.719] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696922.720337447] [grunt1.agent.ros_ai_bridge]: [08:35:22.719] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.720977897] [grunt1.agent.ros_ai_bridge]: [08:35:22.720] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696922.721477151] [grunt1.agent.simple_audio_player]: Starting playback, queue size: 1
[simple_audio_player-2] [WARN] [1755696922.722036832] [grunt1.agent.simple_audio_player]: Not playing, chunk dropped
[voice_chunk_recorder-6] [INFO] [1755696922.723096538] [grunt1.agent.voice_recorder_output]: Started recording audio_out -> /tmp/voice_chunks/assistant_output/audio_out_20250820_083522.wav at 16000 Hz
[simple_audio_player-2] [INFO] [1755696922.726107088] [grunt1.agent.simple_audio_player]: Started audio playback - Assistant speaking
[simple_audio_player-2] [INFO] [1755696922.726613367] [grunt1.agent.simple_audio_player]: First audio chunk received! Size: 1600 samples (0.100s)
[simple_audio_player-2] [INFO] [1755696922.727052347] [grunt1.agent.simple_audio_player]: Queue empty: True, Playing: True
[oai_realtime_agent-7] [08:35:22.761] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:22.763] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.765079629] [grunt1.agent.ros_ai_bridge]: [08:35:22.764] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696922.767766878] [grunt1.agent.ros_ai_bridge]: [08:35:22.766] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.768872186] [grunt1.agent.ros_ai_bridge]: [08:35:22.767] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696922.769489534] [grunt1.agent.ros_ai_bridge]: [08:35:22.768] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.770581433] [grunt1.agent.ros_ai_bridge]: [08:35:22.769] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696922.771303354] [grunt1.agent.simple_audio_player]: Audio chunk 2: 2400 samples
[simple_audio_player-2] [INFO] [1755696922.772834213] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 2400 samples
[ros_ai_bridge-4] [INFO] [1755696922.800707745] [grunt1.agent.ros_ai_bridge]: [08:35:22.800] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 8
[ros_ai_bridge-4] [INFO] [1755696922.801450713] [grunt1.agent.ros_ai_bridge]: [08:35:22.800] [bridge] WebSocket agents: 2 connected
[oai_realtime_agent-7] [08:35:22.834] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:22.837] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.840008418] [grunt1.agent.ros_ai_bridge]: [08:35:22.839] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696922.843649221] [grunt1.agent.ros_ai_bridge]: [08:35:22.841] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.845197854] [grunt1.agent.ros_ai_bridge]: [08:35:22.843] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696922.845429995] [grunt1.agent.ros_ai_bridge]: [08:35:22.844] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.846875278] [grunt1.agent.ros_ai_bridge]: [08:35:22.846] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696922.847154646] [grunt1.agent.simple_audio_player]: Audio chunk 3: 4000 samples
[oai_realtime_agent-7] [08:35:22.938] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:22.940] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.944592230] [grunt1.agent.ros_ai_bridge]: [08:35:22.943] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696922.946809163] [grunt1.agent.ros_ai_bridge]: [08:35:22.945] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.947429361] [grunt1.agent.ros_ai_bridge]: [08:35:22.946] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696922.948409263] [grunt1.agent.ros_ai_bridge]: [08:35:22.947] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696922.949121910] [grunt1.agent.ros_ai_bridge]: [08:35:22.948] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696922.949623810] [grunt1.agent.simple_audio_player]: Audio chunk 4: 4000 samples
[simple_audio_player-2] [INFO] [1755696922.950952226] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 4000 samples
[oai_realtime_agent-7] [08:35:22.997] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[ros_ai_bridge-4] [INFO] [1755696923.002083695] [grunt1.agent.ros_ai_bridge]: [08:35:23.001] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[oai_realtime_agent-7] [08:35:23.003] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.014558879] [grunt1.agent.ros_ai_bridge]: [08:35:23.002] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.016856687] [grunt1.agent.ros_ai_bridge]: [08:35:23.015] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696923.017084620] [grunt1.agent.ros_ai_bridge]: [08:35:23.015] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.017867207] [grunt1.agent.ros_ai_bridge]: [08:35:23.017] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696923.020153986] [grunt1.agent.simple_audio_player]: Audio chunk 5: 4000 samples
[oai_realtime_agent-7] [08:35:23.172] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:23.174] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.176761622] [grunt1.agent.ros_ai_bridge]: [08:35:23.176] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696923.178070707] [grunt1.agent.ros_ai_bridge]: [08:35:23.177] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.178791679] [grunt1.agent.ros_ai_bridge]: [08:35:23.178] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696923.181085224] [grunt1.agent.ros_ai_bridge]: [08:35:23.180] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.183701401] [grunt1.agent.ros_ai_bridge]: [08:35:23.182] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696923.183894620] [grunt1.agent.simple_audio_player]: Audio chunk 6: 4000 samples
[simple_audio_player-2] [INFO] [1755696923.195298330] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 4000 samples
[oai_realtime_agent-7] [08:35:23.212] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:23.214] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.218138934] [grunt1.agent.ros_ai_bridge]: [08:35:23.216] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696923.220324823] [grunt1.agent.ros_ai_bridge]: [08:35:23.218] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.220768767] [grunt1.agent.ros_ai_bridge]: [08:35:23.220] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696923.221385446] [grunt1.agent.ros_ai_bridge]: [08:35:23.220] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.221984728] [grunt1.agent.ros_ai_bridge]: [08:35:23.221] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696923.222399836] [grunt1.agent.simple_audio_player]: Audio chunk 7: 4000 samples
[oai_realtime_agent-7] [08:35:23.264] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:23.266] [agent:conv] ✅ Successfully sent message to topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.269331539] [grunt1.agent.ros_ai_bridge]: [08:35:23.267] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696923.271678126] [grunt1.agent.ros_ai_bridge]: [08:35:23.270] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.272526495] [grunt1.agent.ros_ai_bridge]: [08:35:23.271] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696923.272913189] [grunt1.agent.ros_ai_bridge]: [08:35:23.271] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.273564248] [grunt1.agent.ros_ai_bridge]: [08:35:23.273] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696923.274349801] [grunt1.agent.simple_audio_player]: Audio chunk 8: 4000 samples
[oai_realtime_agent-7] [08:35:23.365] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "audio_out", "msg_type": "audio_common_msgs/AudioData", "data"...
[oai_realtime_agent-7] [08:35:23.369] [agent:conv] ✅ Successfully sent message to topic: audio_out
[oai_realtime_agent-7] [08:35:23.369] [agent:conv] 🎯 OpenAI: response.audio.done
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🎯 OpenAI: response.audio_transcript.done
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🤖 Assistant transcript: Good morning! How can I help you today?
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "llm_transcript", "msg_type": "std_msgs/String", "data": {"dat...
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] ✅ Successfully sent message to topic: llm_transcript
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 📤 Assistant transcript sent to ROS
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🎯 OpenAI: response.content_part.done
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🎯 OpenAI: response.output_item.done
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🎯 OpenAI: response.done
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🤖 Assistant response complete
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] ✅ audio_complete complete
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🔄 All responses complete - ready to cycle session
[oai_realtime_agent-7] [08:35:23.370] [agent:conv] 🎯 OpenAI: rate_limits.updated
[ros_ai_bridge-4] [INFO] [1755696923.371879358] [grunt1.agent.ros_ai_bridge]: [08:35:23.370] [bridge] 📥 Received outbound message: topic=audio_out, type=audio_common_msgs/AudioData
[ros_ai_bridge-4] [INFO] [1755696923.374545132] [grunt1.agent.ros_ai_bridge]: [08:35:23.373] [bridge] 📤 Queueing outbound message for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.375387626] [grunt1.agent.ros_ai_bridge]: [08:35:23.374] [bridge] ✅ Successfully queued outbound message for audio_out
[ros_ai_bridge-4] [INFO] [1755696923.376471640] [grunt1.agent.ros_ai_bridge]: [08:35:23.375] [bridge] 📤 Processing outbound message from queue for topic: audio_out
[ros_ai_bridge-4] [INFO] [1755696923.377667572] [grunt1.agent.ros_ai_bridge]: [08:35:23.376] [bridge] ✅ Published message to ROS topic: /audio_out
[simple_audio_player-2] [INFO] [1755696923.379988954] [grunt1.agent.simple_audio_player]: Audio chunk 9: 9600 samples
[ros_ai_bridge-4] [INFO] [1755696923.380690307] [grunt1.agent.ros_ai_bridge]: [08:35:23.375] [bridge] 📥 Received outbound message: topic=llm_transcript, type=std_msgs/String
[ros_ai_bridge-4] [INFO] [1755696923.381277641] [grunt1.agent.ros_ai_bridge]: [08:35:23.380] [bridge] 📤 Queueing outbound message for topic: llm_transcript
[ros_ai_bridge-4] [INFO] [1755696923.382459123] [grunt1.agent.ros_ai_bridge]: [08:35:23.381] [bridge] ✅ Successfully queued outbound message for llm_transcript
[ros_ai_bridge-4] [INFO] [1755696923.383414616] [grunt1.agent.ros_ai_bridge]: [08:35:23.381] [bridge] 📤 Processing outbound message from queue for topic: llm_transcript
[ros_ai_bridge-4] [INFO] [1755696923.384227036] [grunt1.agent.ros_ai_bridge]: [08:35:23.383] [bridge] ✅ Published message to ROS topic: /llm_transcript
[simple_audio_player-2] [INFO] [1755696923.436272491] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 4000 samples
[oai_realtime_agent-7] [08:35:23.457] [agent:conv] 🎯 OpenAI: conversation.item.input_audio_transcription.delta
[oai_realtime_agent-7] [08:35:23.457] [agent:conv] 🎯 OpenAI: conversation.item.input_audio_transcription.completed
[oai_realtime_agent-7] [08:35:23.458] [agent:conv] 👤 User transcript: Good morning.
[oai_realtime_agent-7] [08:35:23.458] [agent:conv] ✅ transcription complete
[oai_realtime_agent-7] [08:35:23.458] [agent:conv] 🔄 All responses complete - ready to cycle session
[simple_audio_player-2] [INFO] [1755696923.680050321] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 4000 samples
[ros_ai_bridge-4] [INFO] [1755696923.798871288] [grunt1.agent.ros_ai_bridge]: [08:35:23.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 16
[ros_ai_bridge-4] [INFO] [1755696923.799391713] [grunt1.agent.ros_ai_bridge]: [08:35:23.798] [bridge] WebSocket agents: 2 connected
[simple_audio_player-2] [INFO] [1755696923.998578674] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 4000 samples
[simple_audio_player-2] [INFO] [1755696924.238560499] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 4000 samples
[ros_ai_bridge-4] [INFO] [1755696924.799100653] [grunt1.agent.ros_ai_bridge]: [08:35:24.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 16
[ros_ai_bridge-4] [INFO] [1755696924.799699663] [grunt1.agent.ros_ai_bridge]: [08:35:24.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696924.804594104] [grunt1.agent.silero_vad_node]: [08:35:24.803] [vad] Audio chunk #1800: 512 samples
[silero_vad_node-3] [DEBUG] [1755696924.807013606] [grunt1.agent.silero_vad_node]: [08:35:24.806] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[simple_audio_player-2] [INFO] [1755696924.807165750] [grunt1.agent.simple_audio_player]: Played audio chunk, size: 9600 samples
[ros_ai_bridge-4] [INFO] [1755696925.798955577] [grunt1.agent.ros_ai_bridge]: [08:35:25.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 16
[ros_ai_bridge-4] [INFO] [1755696925.799479635] [grunt1.agent.ros_ai_bridge]: [08:35:25.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696926.798967351] [grunt1.agent.ros_ai_bridge]: [08:35:26.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 16
[ros_ai_bridge-4] [INFO] [1755696926.799491708] [grunt1.agent.ros_ai_bridge]: [08:35:26.799] [bridge] WebSocket agents: 2 connected
[ros_ai_bridge-4] [INFO] [1755696927.799586345] [grunt1.agent.ros_ai_bridge]: [08:35:27.798] [bridge] Bridge metrics - Inbound: 4, Outbound: 0, Dropped: 0, Total: 16
[ros_ai_bridge-4] [INFO] [1755696927.801847084] [grunt1.agent.ros_ai_bridge]: [08:35:27.799] [bridge] WebSocket agents: 2 connected
[silero_vad_node-3] [INFO] [1755696928.018646658] [grunt1.agent.silero_vad_node]: [08:35:28.018] [vad] Audio chunk #1900: 512 samples
[silero_vad_node-3] [DEBUG] [1755696928.020932801] [grunt1.agent.silero_vad_node]: [08:35:28.020] [vad] DEBUG: Processed 512-sample VAD chunk, 0 samples remaining in buffer
[oai_realtime_agent-8] [08:35:28.065] [agent:cmd] 🎯 OpenAI: conversation.item.input_audio_transcription.delta
[oai_realtime_agent-8] [08:35:28.065] [agent:cmd] 🎯 OpenAI: conversation.item.input_audio_transcription.completed
[oai_realtime_agent-8] [08:35:28.065] [agent:cmd] 👤 User transcript: Good morning.
[oai_realtime_agent-8] [08:35:28.065] [agent:cmd] ✅ transcription complete
[oai_realtime_agent-8] [08:35:28.065] [agent:cmd] 🤖 Triggering OpenAI response generation
[oai_realtime_agent-8] [08:35:28.066] [agent:cmd] ✅ Response generation triggered
[oai_realtime_agent-8] [08:35:28.111] [agent:cmd] 🎯 OpenAI: response.created
[oai_realtime_agent-8] [08:35:28.111] [agent:cmd] 🤖 OpenAI creating response...
[oai_realtime_agent-8] [08:35:28.111] [agent:cmd] ✅ assistant_response complete
[oai_realtime_agent-8] [08:35:28.514] [agent:cmd] 🎯 OpenAI: response.output_item.added
[oai_realtime_agent-8] [08:35:28.514] [agent:cmd] 🎯 OpenAI: conversation.item.created
[oai_realtime_agent-8] [08:35:28.514] [agent:cmd] 🤖 Assistant starting response (item: item_C6dGKq9aCdVIbg1XLA1S3)
[oai_realtime_agent-8] [08:35:28.537] [agent:cmd] 🎯 OpenAI: response.content_part.added
[silero_vad_node-3] [INFO] [1755696928.705160408] [grunt1.agent.silero_vad_node]: [08:35:28.704] [vad] Voice activity: True
[silero_vad_node-3] [INFO] [1755696928.706559482] [grunt1.agent.silero_vad_node]: [08:35:28.706] [vad] Voice detected. Starting utterance 1755696928661666048.
[silero_vad_node-3] [INFO] [1755696928.707215486] [grunt1.agent.silero_vad_node]: [08:35:28.706] [vad] Initialized chunking buffer with 15 pre-roll frames
[oai_realtime_agent-8] [08:35:28.769] [agent:cmd] 🎯 OpenAI: response.audio.done
[oai_realtime_agent-8] [08:35:28.769] [agent:cmd] 🎯 OpenAI: response.audio_transcript.done
[oai_realtime_agent-8] [08:35:28.769] [agent:cmd] 🤖 Assistant transcript: Good morning!
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 📤 Sending outbound message to bridge: {"type": "outbound_message", "topic": "command_transcript", "msg_type": "std_msgs/String", "data": {...
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] ✅ Successfully sent message to topic: command_transcript
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 📤 Assistant transcript sent to ROS
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 🎯 OpenAI: response.content_part.done
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 🎯 OpenAI: response.output_item.done
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 🎯 OpenAI: response.done
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 🤖 Assistant response complete
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] ✅ audio_complete complete
[oai_realtime_agent-8] [08:35:28.770] [agent:cmd] 🔄 All responses complete - ready to cycle session
[ros_ai_bridge-4] [INFO] [1755696928.771441078] [grunt1.agent.ros_ai_bridge]: [08:35:28.770] [bridge] 📥 Received outbound message: topic=command_transcript, type=std_msgs/String
[ros_ai_bridge-4] [INFO] [1755696928.772006891] [grunt1.agent.ros_ai_bridge]: [08:35:28.771] [bridge] 📤 Queueing outbound message for topic: command_transcript
[ros_ai_bridge-4] [INFO] [1755696928.772505646] [grunt1.agent.ros_ai_bridge]: [08:35:28.772] [bridge] ✅ Successfully queued outbound message for command_transcript
[ros_ai_bridge-4] [INFO] [1755696928.773197158] [grunt1.agent.ros_ai_bridge]: [08:35:28.772] [bridge] 📤 Processing outbound message from queue for topic: command_transcript
[ros_ai_bridge-4] [INFO] [1755696928.773787742] [grunt1.agent.ros_ai_bridge]: [08:35:28.773] [bridge] ✅ Published message to ROS topic: /command_transcript
[command_processor-5] [INFO] [1755696928.774271083] [grunt1.agent.command_processor]: Received command: 'Good morning!'
[command_processor-5] [WARN] [1755696928.774806969] [grunt1.agent.command_processor]: Unrecognized command: 'good morning!'
[oai_realtime_agent-8] [08:35:28.789] [agent:cmd] 🎯 OpenAI: rate_limits.updated


## System Startup to First Response - Detailed Event Trace

This document analyzes the successful operation of the OpenAI dual agent system, documenting every event from startup through the first complete user-assistant interaction.

## Configuration Context
- **Launch**: `oai_dual_agent.launch.py`
- **Namespace**: `grunt1`
- **Prefix**: `agent`
- **Two agents running**:
  - Conversational Agent (conv)
  - Command Extraction Agent (cmd)
- **Model**: `gpt-4o-realtime-preview`
- **Voices**: `echo` (conv), `alloy` (cmd)

## Phase 1: Process Startup (T+0.000s - T+1.000s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+0.000 | launch | Process start | launch.py | 7 processes started with PIDs |
| T+0.500 | audio_capturer | Node started | - | ALSA initialization warnings (normal) |
| T+0.525 | ros_ai_bridge | Config loaded | ros_ai_bridge.py | Loaded bridge_dual_agent.yaml |
| T+0.527 | ros_ai_bridge | Topics setup | ros_ai_bridge.py | 3 subscriptions, 6 publishers configured |
| T+0.530 | voice_recorder | Started | - | Recording audio_out at 16000 Hz |
| T+0.794 | ros_ai_bridge | WebSocket ready | ros_ai_bridge.py | Server started on 0.0.0.0:8765 |
| T+0.833 | oai_agent (conv) | Initialized | oai_realtime_agent.py:__init__ | Conversation monitor started |
| T+0.833 | oai_agent (cmd) | Initialized | oai_realtime_agent.py:__init__ | Command extractor initialized |

## Phase 2: Agent-Bridge Connection (T+1.000s - T+6.000s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+0.833 | oai_agent (conv) | Connection attempt | oai_realtime_agent.py:_connect_to_bridge | Connecting to ws://localhost:8765 |
| T+0.854 | ros_ai_bridge | Accept connection | ros_ai_bridge.py | Connection from 127.0.0.1:39110 |
| T+0.855 | oai_agent (conv) | Registered | oai_realtime_agent.py:_connect_to_bridge | Session: sess_openai_realtime_1755696865 |
| T+0.856 | ros_ai_bridge | Agent registered | ros_ai_bridge.py | Capabilities: ['audio_processing', 'realtime_api'] |
| T+0.857 | oai_agent (conv) | Publish conv ID | oai_realtime_agent.py:run | Published: conv_20250820_083425_833244 |
| T+0.875 | oai_agent (conv) | Process conv ID | oai_realtime_agent.py:_handle_conversation_id_message | Creating session for incoming message |
| T+0.876 | oai_agent (conv) | OpenAI connect | oai_session_manager.py:connect_session | URL: wss://api.openai.com/v1/realtime |
| T+5.405 | oai_agent (cmd) | Connection retry | oai_realtime_agent.py:_connect_to_bridge | Initial connection failed (DNS) |
| T+5.410 | oai_agent (cmd) | Registered | oai_realtime_agent.py:_connect_to_bridge | Session: sess_openai_command_extractor |
| T+6.427 | oai_agent (cmd) | Process conv ID | oai_realtime_agent.py:_handle_conversation_id_message | Creating session for incoming message |

## Phase 3: DNS Resolution Issues & Recovery (T+21.897s - T+53.911s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+21.897 | oai_agent (conv) | DNS fail | oai_session_manager.py:connect_session | [Errno -3] Temporary failure in name resolution |
| T+21.908 | oai_agent (conv) | Retry | oai_realtime_agent.py:_ensure_session_for_message | Processing queued conversation_id message |
| T+26.448 | oai_agent (cmd) | DNS fail | oai_session_manager.py:connect_session | [Errno -3] Temporary failure in name resolution |
| T+26.459 | oai_agent (cmd) | Process audio | oai_realtime_agent.py:_handle_audio_message | Processing queued AudioDataUtterance |
| T+40.816 | Both agents | DNS fail | oai_session_manager.py:connect_session | Both agents fail again |
| T+40.827 | oai_agent (conv) | Process audio | oai_realtime_agent.py:_handle_audio_message | Processing second AudioDataUtterance |
| T+53.911 | oai_agent (conv) | Connected | oai_session_manager.py:connect_session | WebSocket connected successfully |
| T+53.913 | oai_agent (cmd) | Connected | oai_session_manager.py:connect_session | WebSocket connected successfully |

## Phase 4: Session Configuration (T+53.911s - T+54.152s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+53.911 | oai_agent (conv) | Config sent | oai_session_manager.py:_configure_session | Voice: echo, VAD enabled |
| T+53.913 | oai_agent (cmd) | Config sent | oai_session_manager.py:_configure_session | Voice: alloy, VAD enabled |
| T+53.914 | Both agents | Session created | oai_session_manager.py:connect_session | Sessions active, response processors started |
| T+53.925 | oai_agent (conv) | Audio sent | oai_serializer.py:serialize_audio | Chunk #0: 68318 bytes |
| T+53.930 | oai_agent (cmd) | Audio sent | oai_serializer.py:serialize_audio | Chunk #0: 69682 bytes |
| T+54.028 | oai_agent (conv) | Buffer commit | oai_session_manager.py:_handle_server_event | Audio committed for utterance |
| T+54.034 | oai_agent (cmd) | Buffer commit | oai_session_manager.py:_handle_server_event | Audio committed for utterance |
| T+54.152 | oai_agent (conv) | Second audio | oai_serializer.py:serialize_audio | Processing second utterance |

## Phase 5: OpenAI Processing (T+55.477s - T+57.370s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+55.477 | OpenAI | session.updated | oai_session_manager.py:_handle_server_event | Server VAD configured |
| T+55.515 | OpenAI | speech_started | oai_session_manager.py:_handle_server_event | VAD detected speech |
| T+55.575 | OpenAI | buffer.committed | oai_session_manager.py:_handle_server_event | First audio buffer committed |
| T+55.582 | OpenAI | transcription | oai_session_manager.py:_handle_server_event | Empty transcript (noise) |
| T+55.596 | OpenAI | speech_started | oai_session_manager.py:_handle_server_event | Second speech detection |
| T+57.252 | OpenAI | transcription | oai_session_manager.py:_handle_server_event | "Good morning." transcribed |
| T+57.253 | oai_agent (conv) | Trigger response | oai_realtime_agent.py:_trigger_openai_response | Manual response trigger |
| T+57.298 | OpenAI | response.created | oai_session_manager.py:_handle_server_event | Response generation started |
| T+57.370 | OpenAI | response.done | oai_session_manager.py:_handle_server_event | "Good morning! How can I help you today?" |

## Phase 6: Audio Output (T+57.714s - T+59.807s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+57.714 | oai_agent (conv) | Audio chunk 1 | oai_realtime_agent.py:_handle_audio_delta | 1600 samples sent |
| T+57.717 | ros_ai_bridge | Publish audio | ros_ai_bridge.py | Published to /audio_out |
| T+57.722 | simple_audio_player | Start playback | - | Assistant speaking |
| T+57.761-58.365 | oai_agent (conv) | Audio chunks 2-9 | oai_realtime_agent.py:_handle_audio_delta | 2400-9600 samples per chunk |
| T+58.370 | oai_agent (conv) | Transcript sent | oai_realtime_agent.py:_handle_transcript_delta | Published to /llm_transcript |
| T+59.807 | simple_audio_player | Playback complete | - | All audio chunks played |

## Phase 7: Command Extractor Response (T+63.065s - T+63.774s)

| Time | Component | Event | Code Reference | Description |
|------|-----------|-------|----------------|-------------|
| T+63.065 | oai_agent (cmd) | Transcription | oai_session_manager.py:_handle_server_event | "Good morning." transcribed |
| T+63.066 | oai_agent (cmd) | Trigger response | oai_realtime_agent.py:_trigger_openai_response | Manual response trigger |
| T+63.111 | OpenAI | response.created | oai_session_manager.py:_handle_server_event | Command extraction started |
| T+63.769 | OpenAI | response.done | oai_session_manager.py:_handle_server_event | "Good morning!" extracted |
| T+63.770 | oai_agent (cmd) | Publish command | oai_realtime_agent.py:_handle_transcript_delta | Published to /command_transcript |
| T+63.774 | command_processor | Process command | - | Unrecognized: 'good morning!' |

## Key Observations

### Successful Patterns
1. **Bridge Connection**: Agents connect to bridge WebSocket immediately after startup
2. **Registration**: Agents register with bridge and subscribe to topics
3. **Conversation ID**: Published immediately after connection to establish context
4. **Session Creation**: Triggered by first incoming message (conversation_id or audio)
5. **DNS Recovery**: System successfully recovers from DNS failures with retry logic
6. **Audio Pipeline**: Bridge serializes audio, agents process, OpenAI responds
7. **Response Flow**: Transcription → Manual trigger → Response generation → Audio output

### Critical Sequencing
1. Bridge must be ready before agents connect (WebSocket on port 8765)
2. Agents must register and subscribe before receiving messages
3. OpenAI session must be created before processing audio
4. Response processor task starts immediately after session creation
5. Audio chunks are sent sequentially with proper buffering

### Error Recovery
- DNS failures handled gracefully with retry logic
- Queued messages processed after connection established
- Multiple connection attempts with exponential backoff

## Comparison Points for Gemini Implementation

The Gemini agent must replicate this exact sequence:
1. Connect to bridge WebSocket at startup
2. Register with appropriate capabilities
3. Subscribe to required topics
4. Publish conversation ID immediately
5. Create session on first message
6. Start response processor task
7. Process audio/text messages
8. Publish responses to correct topics
9. Handle connection failures gracefully
10. Maintain non-blocking async operations