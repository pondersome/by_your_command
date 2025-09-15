graph [
  version 2
  directed 1
  compound "true"
  fontname "Arial"
  rankdir "TB"
  node [
    id 0
    name "A"
    label "🎤 Microphone"
    graphics [
      hasFill 1
      type "box"
      fill "#2196f3"
    ]
    LabelGraphics [
      text "🎤 Microphone"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 1
    name "B"
    label "Audio Capturer"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Audio Capturer"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 2
    name "C"
    label "📷 Camera"
    graphics [
      hasFill 1
      type "box"
      fill "#2196f3"
    ]
    LabelGraphics [
      text "📷 Camera"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 3
    name "D"
    label "Image Compression"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Image Compression"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 4
    name "E"
    label "Silero VAD"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Silero VAD"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 5
    name "F"
    label "Prompt Voice"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Prompt Voice"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 6
    name "G"
    label "Compressed Images"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Compressed Images"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 7
    name "H"
    label "WebSocket Server\nPort 8765"
    graphics [
      hasFill 1
      type "box"
      fill "#616161"
    ]
    LabelGraphics [
      text "WebSocket Server\nPort 8765"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 8
    name "I"
    label "Prompt Text"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Prompt Text"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 9
    name "J"
    label "Conversational Agent\nbarney_conversational"
    graphics [
      hasFill 1
      type "box"
      fill "#00bcd4"
    ]
    LabelGraphics [
      text "Conversational Agent\nbarney_conversational"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 10
    name "K"
    label "Command Agent\nbarney_command_extractor"
    graphics [
      hasFill 1
      type "box"
      fill "#ff5722"
    ]
    LabelGraphics [
      text "Command Agent\nbarney_command_extractor"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 11
    name "L"
    label "Process for Dialogue"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Process for Dialogue"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 12
    name "M"
    label "Detect but Don't Execute\nCommands"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Detect but Don't Execute\nCommands"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 13
    name "N"
    label "Extract Commands"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Extract Commands"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 14
    name "O"
    label "Generate Scene JSON"
    graphics [
      hasFill 1
      type "box"
      fill "#37474f"
    ]
    LabelGraphics [
      text "Generate Scene JSON"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 15
    name "P"
    label "response_voice\nVoice Response"
    graphics [
      hasFill 1
      type "box"
      fill "#4caf50"
    ]
    LabelGraphics [
      text "response_voice\nVoice Response"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 16
    name "Q"
    label "response_text\nConversation Text"
    graphics [
      hasFill 1
      type "box"
      fill "#4caf50"
    ]
    LabelGraphics [
      text "response_text\nConversation Text"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 17
    name "R"
    label "Brief Acknowledgment\n'OK - moving forward'"
    graphics [
      hasFill 1
      type "box"
      fill "#757575"
    ]
    LabelGraphics [
      text "Brief Acknowledgment\n'OK - moving forward'"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 18
    name "S"
    label "response_cmd\nMovement Commands"
    graphics [
      hasFill 1
      type "box"
      fill "#ff9800"
    ]
    LabelGraphics [
      text "response_cmd\nMovement Commands"
      fontColor "black"
      fontName "Arial"
    ]
  ]
  node [
    id 19
    name "T"
    label "Command Processor"
    graphics [
      hasFill 1
      type "box"
      fill "#388e3c"
    ]
    LabelGraphics [
      text "Command Processor"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  node [
    id 20
    name "U"
    label "Robot Actuators"
    graphics [
      hasFill 1
      type "box"
      fill "#2e7d32"
    ]
    LabelGraphics [
      text "Robot Actuators"
      fontColor "white"
      fontName "Arial"
    ]
  ]
  edge [
    id 1
    source 0
    target 1
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 3
    source 1
    target 4
    lhead ""
    ltail "cluster_input"
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 2
    source 2
    target 3
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 5
    source 3
    target 6
    lhead ""
    ltail "cluster_input"
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 4
    source 4
    target 5
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 7
    source 5
    target 7
    lhead ""
    ltail "cluster_processing"
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 8
    source 6
    target 7
    lhead ""
    ltail "cluster_processing"
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 13
    source 7
    target 9
    lhead "cluster_agents"
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 14
    source 7
    target 10
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 6
    source 8
    target 7
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 9
    source 9
    target 11
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 10
    source 9
    target 12
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 11
    source 10
    target 13
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 12
    source 10
    target 14
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 15
    source 11
    target 15
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 16
    source 11
    target 16
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 17
    source 12
    target 17
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 18
    source 13
    target 18
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 19
    source 14
    target 18
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 20
    source 18
    target 19
    lhead ""
    ltail "cluster_output"
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
  edge [
    id 21
    source 19
    target 20
    lhead ""
    ltail ""
    LabelGraphics [
      fontSize 10
      fontName "Arial"
    ]
  ]
]
