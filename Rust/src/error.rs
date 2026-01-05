//! # Centralized Error Handling
//!
//! Unified error types for the entire crate using `thiserror`.

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for Reagle operations
#[derive(Error, Debug)]
pub enum ReagleError {
    /// I/O errors (file missing, permission denied, read/write failures)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// VCF parsing errors (malformed records, missing fields)
    #[error("VCF error: {message}")]
    Vcf { message: String },

    /// Invalid data errors (sample count mismatch, marker position out of order)
    #[error("Invalid data: {message}")]
    InvalidData { message: String },

    /// Algorithm errors (non-finite probabilities, convergence failures)
    #[error("Algorithm error: {message}")]
    Algorithm { message: String },

    /// Configuration errors (invalid CLI arguments)
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// File not found errors
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },

    /// Parse errors
    #[error("Parse error at line {line}: {message}")]
    Parse { line: usize, message: String },
}

/// Type alias for Results using ReagleError
pub type Result<T> = std::result::Result<T, ReagleError>;

impl ReagleError {
    /// Create a VCF error with a message
    pub fn vcf(message: impl Into<String>) -> Self {
        Self::Vcf {
            message: message.into(),
        }
    }

    /// Create an invalid data error
    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData {
            message: message.into(),
        }
    }

    /// Create an algorithm error
    pub fn algorithm(message: impl Into<String>) -> Self {
        Self::Algorithm {
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a parse error
    pub fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            line,
            message: message.into(),
        }
    }
}

// Convert noodles VCF errors to ReagleError
impl From<noodles::vcf::header::ParseError> for ReagleError {
    fn from(err: noodles::vcf::header::ParseError) -> Self {
        Self::Vcf {
            message: err.to_string(),
        }
    }
}