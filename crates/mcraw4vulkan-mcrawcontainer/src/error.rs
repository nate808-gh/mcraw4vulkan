use thiserror::Error;

#[derive(Debug, Error)]
pub enum McrawContainerError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("frame {0} is out of range")]
    FrameOutOfRange(u32),

    #[error("audio chunk {0} is out of range")]
    AudioChunkOutOfRange(usize),

    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("invalid index: {0}")]
    InvalidIndex(String),

    #[error("invalid payload span: {0}")]
    InvalidPayloadSpan(String),

    #[error("invalid audio buffer size")]
    InvalidAudioBufferSize,
}
