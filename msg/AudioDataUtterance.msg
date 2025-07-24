# Enhanced AudioData with utterance metadata
# Raw audio data in multiple formats
float32[] float32_data
int32[] int32_data
int16[] int16_data
int8[] int8_data 
uint8[] uint8_data

# Utterance metadata
uint64 utterance_id      # Timestamp (nanoseconds) of first frame in utterance
bool is_utterance_end    # True if this is the last chunk in the utterance
uint32 chunk_sequence    # Sequential chunk number within this utterance (0-based)