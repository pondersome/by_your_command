# System Prompt Architecture Diagrams

This document visualizes how system prompts are constructed from reusable components and how the dual-agent architecture processes commands and conversations.

## 1. Prompt Construction Architecture

Shows how macro components are combined to build complete agent prompts:

```mermaid
graph LR
    subgraph "Macro Components (Reusable Building Blocks)"
        M1[robot_name: 'Barney']
        M2[robot_capabilities<br/>- Skid-steer base<br/>- 4DOF arm with camera<br/>- Vision capabilities]
        M3[arm_presets<br/>- bumper, tenhut<br/>- lookup, lookout, reach]
        M4[bearing_presets<br/>- left, right, forward<br/>- back, full-left, etc.]
        M5[motion_commands<br/>- stop, follow, track<br/>- move, turn, pan]
        M6[compound_commands<br/>- move@forward<br/>- pan@left<br/>- follow@person]
        M7[cmd_response<br/>Rules for mapping<br/>user requests to commands]
        M8[visual_cmd_response<br/>JSON format for<br/>scene descriptions]
        M9[visual_convo_response<br/>Natural language<br/>scene descriptions]
        M10[personality_traits<br/>- Friendly but concise<br/>- First-person references]
    end
    
    subgraph "OpenAI Agent Prompts"
        OP1[barney_conversational<br/>Natural dialogue focus]
        OP2[barney_command_extractor<br/>Command extraction only]
    end
    
    subgraph "Gemini Agent Prompts"
        GP1[barney_conversational_gemini<br/>Conversation + vision]
        GP2[barney_command_visual<br/>Combined mode agent]
    end
    
    M1 --> OP1
    M1 --> OP2
    M1 --> GP1
    M1 --> GP2
    M2 --> OP1
    M2 --> OP2
    M2 --> GP1
    M2 --> GP2
    M7 --> OP1
    M7 --> OP2
    M10 --> OP1
    M10 --> GP1
    M3 --> OP2
    M3 --> GP2
    M4 --> OP2
    M4 --> GP2
    M5 --> OP2
    M5 --> GP2
    M6 --> OP2
    M6 --> GP2
    M8 --> GP2
    M9 --> GP1
    
    style M1 fill:#2196f3,color:#fff
    style M2 fill:#2196f3,color:#fff
    style M3 fill:#4caf50,color:#fff
    style M4 fill:#4caf50,color:#fff
    style M5 fill:#4caf50,color:#fff
    style M6 fill:#4caf50,color:#fff
    style M7 fill:#9c27b0,color:#fff
    style M8 fill:#ff9800,color:#000
    style M9 fill:#ff9800,color:#000
    style M10 fill:#e91e63,color:#fff
    style OP1 fill:#00bcd4,color:#fff
    style OP2 fill:#00bcd4,color:#fff
    style GP1 fill:#ff5722,color:#fff
    style GP2 fill:#ff5722,color:#fff
```

## 2. Dual-Agent Processing Architecture

Illustrates how voice and vision inputs flow through the system to both agents:

```mermaid
flowchart TB
    subgraph "Input Sources"
        A[🎤 Microphone] --> B[Audio Capturer]
        C[📷 Camera] --> D[Image Compression]
    end
    
    subgraph "Processing Pipeline"
        B --> E[Silero VAD]
        E --> F[Voice Chunks]
        D --> G[Compressed Images]
    end
    
    subgraph "ROS AI Bridge"
        F --> H[WebSocket Server<br/>Port 8765]
        G --> H
        I[Text Input] --> H
    end
    
    subgraph "Parallel Agent Processing"
        H --> J[Conversational Agent<br/>barney_conversational]
        H --> K[Command Agent<br/>barney_command_extractor]
        
        J --> L[Process for Dialogue]
        J --> M[Detect but Don't Execute Commands]
        
        K --> N[Extract Commands]
        K --> O[Generate Scene JSON]
    end
    
    subgraph "Output Topics"
        L --> P[response_voice<br/>Voice Response]
        L --> Q[response_text<br/>Conversation Text]
        M --> R["Brief Acknowledgment<br/>'OK - moving forward'"]
        N --> S[response_cmd<br/>Movement Commands]
        O --> S
    end
    
    subgraph "Robot Control"
        S --> T[Command Processor]
        T --> U[Robot Actuators]
    end
    
    style A fill:#2196f3,color:#fff
    style C fill:#2196f3,color:#fff
    style J fill:#00bcd4,color:#fff
    style K fill:#ff5722,color:#fff
    style P fill:#4caf50,color:#fff
    style Q fill:#4caf50,color:#fff
    style S fill:#ff9800,color:#000
```

## 3. Agent Behavior State Machines

Shows the decision logic for each agent type:

```mermaid
stateDiagram-v2
    [*] --> InputProcessing
    
    state "Shared Input Processing" as InputProcessing {
        [*] --> Waiting
        Waiting --> ReceiveInput: Bridge Message
        
        state ReceiveInput {
            Voice: Voice chunks
            Text: Text input
            Camera: Camera frames
            Context: Conversation cache
        }
        
        ReceiveInput --> ProcessWithLLM: Send to LLM
    }
    
    ProcessWithLLM --> AgentBehaviors
    
    state "Agent-Specific Behaviors" as AgentBehaviors {
        state fork_state <<fork>>
        ProcessWithLLM --> fork_state
        
        state "Conversational Agent" as Conv {
            fork_state --> AnalyzeIntent
            
            AnalyzeIntent --> DetectedCommand: Is command?
            AnalyzeIntent --> NormalConvo: Is conversation?
            AnalyzeIntent --> VisionQuery: Is vision query?
            
            DetectedCommand --> BriefAck: Acknowledge only
            BriefAck --> Output1: OK - moving forward
            
            NormalConvo --> FullResponse: Generate dialogue
            FullResponse --> Output2: Natural conversation
            
            VisionQuery --> DescribeScene: Generate description  
            DescribeScene --> Output3: Natural language scene
            
            Output1 --> PublishTranscript: To response_text
            Output2 --> PublishTranscript
            Output3 --> PublishTranscript

            Output1 --> PublishAudio: To response_voice
            Output2 --> PublishAudio
            Output3 --> PublishAudio
        }
        
        state "Command Agent" as Cmd {
            fork_state --> EvaluateInput
            
            EvaluateInput --> IsCommand: Command detected?
            EvaluateInput --> IsVision: Vision request?
            EvaluateInput --> NotRelevant: Neither?
            
            IsCommand --> ExtractCmd: Parse command
            ExtractCmd --> FormatCmd: Format as text
            FormatCmd --> OutputCmd: bumper@left
            
            IsVision --> ExtractScene: Analyze image
            ExtractScene --> FormatJSON: Create JSON
            FormatJSON --> OutputScene: Scene JSON
            
            NotRelevant --> Silent: No output
            
            OutputCmd --> PublishCmd: To response_cmd
            OutputScene --> PublishCmd
        }
    }
```

## 4. Prompt Macro Expansion Process

Demonstrates the recursive expansion of macros in prompt templates:

```mermaid
graph TD
    subgraph "Original Template"
        T["barney_conversational prompt template"]
        T --> T1["Your name is {{robot_name}}"]
        T --> T2["{{robot_capabilities}}"]
        T --> T3["{{cmd_response}}"]
        T --> T4["{{personality_traits}}"]
    end
    
    subgraph "First Level Expansion"
        T1 --> E1["Your name is Barney"]
        T2 --> E2["You are a real skid-steer robot<br/>that can move around and you have<br/>one 4dof arm that can move a camera..."]
        T3 --> E3["Infer or use synonyms to map user<br/>requests to valid COMMANDS...<br/>{{arm_presets}}<br/>{{bearing_presets}}<br/>{{motion_commands}}"]
        T4 --> E4["You are friendly but concise.<br/>Provide information and assistance."]
    end
    
    subgraph "Second Level Expansion (Recursive)"
        E3 --> F1["Arm Positions:<br/>bumper - arm folded low<br/>tenhut - vertical stance<br/>lookup - raised to face person<br/>lookout - elevated forward<br/>reach - extended forward"]
        E3 --> F2["Bearing Presets:<br/>left: -45°<br/>right: 45°<br/>forward: 0°<br/>back: 180°<br/>full-left: -90°"]
        E3 --> F3["Motion Commands:<br/>stop, follow, track<br/>sleep, wake, move<br/>turn, pan"]
    end
    
    subgraph "Final Expanded Prompt"
        G["Complete system prompt with<br/>all macros fully expanded<br/>~2000 tokens"]
    end
    
    E1 --> G
    E2 --> G
    E4 --> G
    F1 --> G
    F2 --> G
    F3 --> G
    
    style T fill:#9c27b0,color:#fff
    style T1 fill:#2196f3,color:#fff
    style T2 fill:#2196f3,color:#fff
    style T3 fill:#2196f3,color:#fff
    style T4 fill:#2196f3,color:#fff
    style E3 fill:#ff9800,color:#000
    style F1 fill:#4caf50,color:#fff
    style F2 fill:#4caf50,color:#fff
    style F3 fill:#4caf50,color:#fff
    style G fill:#00bcd4,color:#fff
```

## 5. Command Processing Pipeline

Shows the complete flow from user input to robot action:

```mermaid
sequenceDiagram
    participant User
    participant VAD as Silero VAD
    participant Bridge as ROS AI Bridge
    participant ConvAgent as Conversational<br/>Agent
    participant CmdAgent as Command<br/>Agent
    participant CmdProc as Command<br/>Processor
    participant Robot
    
    User->>VAD: "Look up and move forward"
    VAD->>Bridge: Voice chunks + utterance metadata
    
    par Parallel Processing
        Bridge->>ConvAgent: Voice + Camera image
        Note over ConvAgent: Detects command<br/>but doesn't execute
        ConvAgent->>Bridge: Audio: "OK - looking up and moving"
        ConvAgent->>Bridge: Text: "OK - looking up and moving forward"
    and
        Bridge->>CmdAgent: Voice + Camera image
        Note over CmdAgent: Extracts commands
        CmdAgent->>Bridge: {"command": "ARM:lookup"}
        CmdAgent->>Bridge: {"command": "move@forward"}
    end
    
    Bridge->>CmdProc: Commands via response_cmd
    Bridge->>User: Audio response via response_voice
    
    CmdProc->>Robot: ARM:lookup preset
    CmdProc->>Robot: move@forward command
    Robot->>Robot: Execute movements
    
    Note over User: Hears acknowledgment<br/>while robot moves
```

## 6. Agent Comparison Matrix

| Aspect | Conversational Agent | Command Agent |
|--------|---------------------|---------------|
| **Primary Role** | Natural dialogue & descriptions | Command extraction & scene analysis |
| **Prompt Used (OAI)** | `barney_conversational` | `barney_command_extractor` |
| **Prompt Used (Gemini)** | `barney_conversational_gemini` | `barney_command_visual` |
| **Command Handling** | Acknowledges with "OK - [action]" | Extracts and formats as JSON |
| **Conversation** | Full natural responses | Silent (no response) |
| **Vision Queries** | Natural language descriptions | JSON object list with coordinates |
| **Audio Output** | ✅ Enabled | ❌ Disabled |
| **Text Output Topic** | `/response_text` | `/response_cmd` |
| **Macros Used** | `robot_name`, `robot_capabilities`, `cmd_response`, `personality_traits` | `robot_name`, `robot_capabilities`, `arm_presets`, `bearing_presets`, `motion_commands` |
| **Response Examples** | "I can see a red ball on the table" | `{"command": "ARM:lookup"}` |
| **Silent Conditions** | Never (always responds) | Non-command inputs |

## 7. System Integration Overview

```mermaid
graph LR
    subgraph "Launch Configuration"
        L1[oai_dual_agent.launch.py]
        L2[gemini_dual_agent.launch.py]
    end
    
    subgraph "Configuration Files"
        C1[prompts.yaml<br/>Macro definitions]
        C2[bridge_dual_agent.yaml<br/>Topic mappings]
        C3[agent configs<br/>Model settings]
    end
    
    subgraph "Shared Components"
        S1[PromptLoader<br/>Macro expansion]
        S2[WebSocketBridge<br/>Communication]
        S3[ConversationMonitor<br/>Context tracking]
    end
    
    subgraph "Active Agents"
        A1[Conversational Agent]
        A2[Command Agent]
    end
    
    L1 --> C1
    L1 --> C2
    L1 --> C3
    L2 --> C1
    L2 --> C2
    L2 --> C3
    
    C1 --> S1
    S1 --> A1
    S1 --> A2
    
    C2 --> S2
    S2 --> A1
    S2 --> A2
    
    S3 --> A1
    S3 --> A2
    
    style L1 fill:#00bcd4,color:#fff
    style L2 fill:#ff5722,color:#fff
    style C1 fill:#ff9800,color:#000
    style S1 fill:#9c27b0,color:#fff
    style A1 fill:#2196f3,color:#fff
    style A2 fill:#4caf50,color:#fff
```

## Key Insights

1. **Macro System**: The prompt construction uses a sophisticated macro expansion system that allows reusable components to be shared across different agent configurations.

2. **Separation of Concerns**: The dual-agent architecture cleanly separates conversation handling from command extraction, allowing each agent to specialize.

3. **Parallel Processing**: Both agents receive the same inputs simultaneously but process them differently based on their specialized prompts.

4. **Graceful Degradation**: If one agent fails, the other can continue operating, providing robustness to the system.

5. **Platform Flexibility**: The same architecture supports both OpenAI and Gemini backends with platform-specific prompt optimizations.