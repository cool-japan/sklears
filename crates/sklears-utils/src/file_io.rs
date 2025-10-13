//! File I/O utilities for efficient data handling in machine learning workflows
//!
//! This module provides utilities for efficient file reading/writing, compression,
//! format conversion, streaming I/O operations, and data serialization.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

// ===== EFFICIENT FILE OPERATIONS =====

/// Efficient file reader with buffering and memory management
pub struct EfficientFileReader {
    reader: BufReader<File>,
    #[allow(dead_code)]
    buffer_size: usize,
}

impl EfficientFileReader {
    /// Create a new efficient file reader
    pub fn new<P: AsRef<Path>>(path: P, buffer_size: Option<usize>) -> UtilsResult<Self> {
        let file = File::open(path)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to open file: {e}")))?;

        let buffer_size = buffer_size.unwrap_or(8192);
        let reader = BufReader::with_capacity(buffer_size, file);

        Ok(Self {
            reader,
            buffer_size,
        })
    }

    /// Read lines efficiently with iterator
    pub fn read_lines(&mut self) -> impl Iterator<Item = UtilsResult<String>> + '_ {
        std::io::BufRead::lines(&mut self.reader).map(|line| {
            line.map_err(|e| UtilsError::InvalidParameter(format!("Failed to read line: {e}")))
        })
    }

    /// Read fixed-size chunks
    pub fn read_chunk(&mut self, size: usize) -> UtilsResult<Vec<u8>> {
        let mut buffer = vec![0u8; size];
        let bytes_read = self
            .reader
            .read(&mut buffer)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read chunk: {e}")))?;

        buffer.truncate(bytes_read);
        Ok(buffer)
    }

    /// Read all content efficiently
    pub fn read_all(&mut self) -> UtilsResult<Vec<u8>> {
        let mut content = Vec::new();
        self.reader
            .read_to_end(&mut content)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read file: {e}")))?;
        Ok(content)
    }

    /// Read numerical data as arrays
    pub fn read_array1(&mut self, delimiter: &str) -> UtilsResult<Array1<f64>> {
        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read line: {e}")))?;

        let values: Result<Vec<f64>, _> = line
            .trim()
            .split(delimiter)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<f64>())
            .collect();

        let values = values
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to parse numbers: {e}")))?;

        Ok(Array1::from_vec(values))
    }

    /// Read 2D numerical data
    pub fn read_array2(&mut self, delimiter: &str) -> UtilsResult<Array2<f64>> {
        let mut rows = Vec::new();
        let mut line = String::new();

        while self.reader.read_line(&mut line).unwrap_or(0) > 0 {
            if line.trim().is_empty() {
                line.clear();
                continue;
            }

            let values: Result<Vec<f64>, _> = line
                .trim()
                .split(delimiter)
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<f64>())
                .collect();

            let values = values.map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to parse numbers: {e}"))
            })?;

            rows.push(values);
            line.clear();
        }

        if rows.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let ncols = rows[0].len();
        let nrows = rows.len();

        // Verify all rows have the same length
        for row in &rows {
            if row.len() != ncols {
                return Err(UtilsError::ShapeMismatch {
                    expected: vec![ncols],
                    actual: vec![row.len()],
                });
            }
        }

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((nrows, ncols), flat)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to create array: {e}")))
    }
}

/// Efficient file writer with buffering
pub struct EfficientFileWriter {
    writer: BufWriter<File>,
}

impl EfficientFileWriter {
    /// Create a new efficient file writer
    pub fn new<P: AsRef<Path>>(path: P, buffer_size: Option<usize>) -> UtilsResult<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to create file: {e}")))?;

        let buffer_size = buffer_size.unwrap_or(8192);
        let writer = BufWriter::with_capacity(buffer_size, file);

        Ok(Self { writer })
    }

    /// Append to existing file
    pub fn append<P: AsRef<Path>>(path: P, buffer_size: Option<usize>) -> UtilsResult<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to open file: {e}")))?;

        let buffer_size = buffer_size.unwrap_or(8192);
        let writer = BufWriter::with_capacity(buffer_size, file);

        Ok(Self { writer })
    }

    /// Write data with automatic flushing
    pub fn write_data(&mut self, data: &[u8]) -> UtilsResult<()> {
        self.writer
            .write_all(data)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write data: {e}")))?;
        Ok(())
    }

    /// Write string lines
    pub fn write_lines<I>(&mut self, lines: I) -> UtilsResult<()>
    where
        I: IntoIterator<Item = String>,
    {
        for line in lines {
            writeln!(self.writer, "{line}")
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write line: {e}")))?;
        }
        Ok(())
    }

    /// Write array data
    pub fn write_array1(&mut self, array: &Array1<f64>, delimiter: &str) -> UtilsResult<()> {
        let line = array
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(delimiter);

        writeln!(self.writer, "{line}")
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write array: {e}")))?;
        Ok(())
    }

    /// Write 2D array data
    pub fn write_array2(&mut self, array: &Array2<f64>, delimiter: &str) -> UtilsResult<()> {
        for row in array.outer_iter() {
            let line = row
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(delimiter);

            writeln!(self.writer, "{line}")
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write row: {e}")))?;
        }
        Ok(())
    }

    /// Flush the buffer
    pub fn flush(&mut self) -> UtilsResult<()> {
        self.writer
            .flush()
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to flush: {e}")))?;
        Ok(())
    }
}

// ===== COMPRESSION UTILITIES =====

/// Simple compression utilities using built-in algorithms
pub struct CompressionUtils;

impl CompressionUtils {
    /// Compress data using flate2 (gzip-compatible)
    #[cfg(feature = "compression")]
    pub fn compress_gzip(data: &[u8]) -> UtilsResult<Vec<u8>> {
        use flate2::{write::GzEncoder, Compression};

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| UtilsError::InvalidParameter(format!("Compression failed: {e}")))?;

        encoder
            .finish()
            .map_err(|e| UtilsError::InvalidParameter(format!("Compression failed: {e}")))
    }

    /// Decompress gzip data
    #[cfg(feature = "compression")]
    pub fn decompress_gzip(data: &[u8]) -> UtilsResult<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder
            .read_to_end(&mut decompressed)
            .map_err(|e| UtilsError::InvalidParameter(format!("Decompression failed: {e}")))?;

        Ok(decompressed)
    }

    /// Simple run-length encoding for sparse data
    pub fn run_length_encode(data: &[u8]) -> Vec<(u8, usize)> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_value = data[0];
        let mut count = 1;

        for &byte in &data[1..] {
            if byte == current_value {
                count += 1;
            } else {
                result.push((current_value, count));
                current_value = byte;
                count = 1;
            }
        }
        result.push((current_value, count));
        result
    }

    /// Decode run-length encoded data
    pub fn run_length_decode(encoded: &[(u8, usize)]) -> Vec<u8> {
        let mut result = Vec::new();
        for &(value, count) in encoded {
            result.extend(std::iter::repeat(value).take(count));
        }
        result
    }
}

// ===== STREAMING I/O =====

/// Streaming data processor for large files
pub struct StreamProcessor<R: Read> {
    reader: R,
    chunk_size: usize,
}

impl<R: Read> StreamProcessor<R> {
    /// Create a new stream processor
    pub fn new(reader: R, chunk_size: usize) -> Self {
        Self { reader, chunk_size }
    }

    /// Process data in chunks with a callback function
    pub fn process_chunks<F>(&mut self, mut processor: F) -> UtilsResult<()>
    where
        F: FnMut(&[u8]) -> UtilsResult<()>,
    {
        let mut buffer = vec![0u8; self.chunk_size];

        loop {
            let bytes_read = self
                .reader
                .read(&mut buffer)
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read chunk: {e}")))?;

            if bytes_read == 0 {
                break;
            }

            processor(&buffer[..bytes_read])?;
        }

        Ok(())
    }

    /// Process lines from a text stream
    pub fn process_lines<F>(&mut self, mut processor: F) -> UtilsResult<()>
    where
        F: FnMut(&str) -> UtilsResult<()>,
        R: BufRead,
    {
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = self
                .reader
                .read_line(&mut line)
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to read line: {e}")))?;

            if bytes_read == 0 {
                break;
            }

            processor(line.trim_end())?;
        }

        Ok(())
    }
}

// ===== FORMAT CONVERSION =====

/// Format conversion utilities
pub struct FormatConverter;

impl FormatConverter {
    /// Convert CSV to structured data
    pub fn csv_to_arrays<P: AsRef<Path>>(
        path: P,
        delimiter: char,
        has_header: bool,
    ) -> UtilsResult<(Option<Vec<String>>, Array2<f64>)> {
        let mut reader = EfficientFileReader::new(path, None)?;
        let mut lines = reader.read_lines();

        let header = if has_header {
            if let Some(line_result) = lines.next() {
                let line = line_result?;
                Some(
                    line.split(delimiter)
                        .map(|s| s.trim().to_string())
                        .collect(),
                )
            } else {
                return Err(UtilsError::EmptyInput);
            }
        } else {
            None
        };

        let mut rows = Vec::new();
        for line_result in lines {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }

            let values: Result<Vec<f64>, _> = line
                .split(delimiter)
                .map(|s| s.trim().parse::<f64>())
                .collect();

            let values = values.map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to parse CSV values: {e}"))
            })?;

            rows.push(values);
        }

        if rows.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let ncols = rows[0].len();
        let nrows = rows.len();

        for row in &rows {
            if row.len() != ncols {
                return Err(UtilsError::ShapeMismatch {
                    expected: vec![ncols],
                    actual: vec![row.len()],
                });
            }
        }

        let flat: Vec<f64> = rows.into_iter().flatten().collect();
        let array = Array2::from_shape_vec((nrows, ncols), flat)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to create array: {e}")))?;

        Ok((header, array))
    }

    /// Convert arrays to CSV format
    pub fn arrays_to_csv<P: AsRef<Path>>(
        path: P,
        data: &Array2<f64>,
        header: Option<&[String]>,
        delimiter: char,
    ) -> UtilsResult<()> {
        let mut writer = EfficientFileWriter::new(path, None)?;

        if let Some(header) = header {
            let header_line = header.join(&delimiter.to_string());
            writeln!(writer.writer, "{header_line}").map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to write header: {e}"))
            })?;
        }

        for row in data.outer_iter() {
            let line = row
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(&delimiter.to_string());

            writeln!(writer.writer, "{line}")
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write row: {e}")))?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Convert JSON-like key-value data
    pub fn json_to_map(json_str: &str) -> UtilsResult<HashMap<String, serde_json::Value>> {
        serde_json::from_str(json_str)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to parse JSON: {e}")))
    }

    /// Convert map to JSON string
    pub fn map_to_json(map: &HashMap<String, serde_json::Value>) -> UtilsResult<String> {
        serde_json::to_string_pretty(map)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to serialize JSON: {e}")))
    }

    /// Convert YAML string to map
    #[cfg(feature = "yaml")]
    pub fn yaml_to_map(yaml_str: &str) -> UtilsResult<HashMap<String, serde_json::Value>> {
        let yaml_value: serde_yaml::Value = serde_yaml::from_str(yaml_str)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to parse YAML: {e}")))?;

        // Convert YAML value to JSON value for consistency
        let json_str = serde_json::to_string(&yaml_value).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to convert YAML to JSON: {e}"))
        })?;

        Self::json_to_map(&json_str)
    }

    /// Convert map to YAML string
    #[cfg(feature = "yaml")]
    pub fn map_to_yaml(map: &HashMap<String, serde_json::Value>) -> UtilsResult<String> {
        serde_yaml::to_string(map)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to serialize YAML: {e}")))
    }

    /// Convert TOML string to map
    #[cfg(feature = "toml_support")]
    pub fn toml_to_map(toml_str: &str) -> UtilsResult<HashMap<String, serde_json::Value>> {
        let toml_value: toml::Value = toml::from_str(toml_str)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to parse TOML: {e}")))?;

        // Convert TOML value to JSON value for consistency
        let json_str = serde_json::to_string(&toml_value).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to convert TOML to JSON: {e}"))
        })?;

        Self::json_to_map(&json_str)
    }

    /// Convert map to TOML string
    #[cfg(feature = "toml_support")]
    pub fn map_to_toml(map: &HashMap<String, serde_json::Value>) -> UtilsResult<String> {
        // Convert JSON values to TOML-compatible values
        let toml_value = serde_json::from_str::<toml::Value>(&serde_json::to_string(map)?)
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to convert to TOML value: {e}"))
            })?;

        toml::to_string_pretty(&toml_value)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to serialize TOML: {e}")))
    }

    /// Convert XML string to simplified map structure
    #[cfg(feature = "xml")]
    pub fn xml_to_simple_map(xml_str: &str) -> UtilsResult<HashMap<String, String>> {
        use quick_xml::events::Event;
        use quick_xml::Reader;

        let mut reader = Reader::from_str(xml_str);
        reader.config_mut().trim_text(true);

        let mut result = HashMap::new();
        let mut current_element = String::new();
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    current_element = String::from_utf8_lossy(e.name().as_ref()).to_string();
                }
                Ok(Event::Text(e)) => {
                    if !current_element.is_empty() {
                        let text = e.unescape().map_err(|e| {
                            UtilsError::InvalidParameter(format!(
                                "Failed to unescape XML text: {}",
                                e
                            ))
                        })?;
                        result.insert(current_element.clone(), text.to_string());
                    }
                }
                Ok(Event::End(_)) => {
                    current_element.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    return Err(UtilsError::InvalidParameter(format!(
                        "Failed to parse XML: {}",
                        e
                    )));
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(result)
    }

    /// Convert map to simple XML structure
    #[cfg(feature = "xml")]
    pub fn simple_map_to_xml(
        map: &HashMap<String, String>,
        root_name: &str,
    ) -> UtilsResult<String> {
        use quick_xml::events::{BytesEnd, BytesStart, BytesText, Event};
        use quick_xml::Writer;
        use std::io::Cursor;

        let mut writer = Writer::new(Cursor::new(Vec::new()));

        // Write XML declaration
        writer
            .write_event(Event::Decl(quick_xml::events::BytesDecl::new(
                "1.0",
                Some("UTF-8"),
                None,
            )))
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to write XML declaration: {e}"))
            })?;

        // Write root element start
        writer
            .write_event(Event::Start(BytesStart::new(root_name)))
            .map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to write root element: {e}"))
            })?;

        // Write map entries
        for (key, value) in map {
            writer
                .write_event(Event::Start(BytesStart::new(key)))
                .map_err(|e| {
                    UtilsError::InvalidParameter(format!("Failed to write element start: {e}"))
                })?;

            writer
                .write_event(Event::Text(BytesText::new(value)))
                .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write text: {e}")))?;

            writer
                .write_event(Event::End(BytesEnd::new(key)))
                .map_err(|e| {
                    UtilsError::InvalidParameter(format!("Failed to write element end: {e}"))
                })?;
        }

        // Write root element end
        writer
            .write_event(Event::End(BytesEnd::new(root_name)))
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to write root end: {e}")))?;

        let result = writer.into_inner().into_inner();
        String::from_utf8(result).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to convert XML to string: {e}"))
        })
    }

    /// Enhanced JSON utilities for ML data structures
    pub fn json_to_arrays(json_str: &str) -> UtilsResult<HashMap<String, Array2<f64>>> {
        let data: HashMap<String, Vec<Vec<f64>>> = serde_json::from_str(json_str).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to parse JSON arrays: {e}"))
        })?;

        let mut result = HashMap::new();
        for (key, matrix) in data {
            if matrix.is_empty() {
                continue;
            }

            let nrows = matrix.len();
            let ncols = matrix[0].len();

            // Verify all rows have the same length
            for row in matrix.iter() {
                if row.len() != ncols {
                    return Err(UtilsError::ShapeMismatch {
                        expected: vec![ncols],
                        actual: vec![row.len()],
                    });
                }
            }

            let flat: Vec<f64> = matrix.into_iter().flatten().collect();
            let array = Array2::from_shape_vec((nrows, ncols), flat).map_err(|e| {
                UtilsError::InvalidParameter(format!("Failed to create array: {e}"))
            })?;

            result.insert(key, array);
        }

        Ok(result)
    }

    /// Convert arrays to JSON format for ML data
    pub fn arrays_to_json(arrays: &HashMap<String, &Array2<f64>>) -> UtilsResult<String> {
        let mut data = HashMap::new();

        for (key, array) in arrays {
            let matrix: Vec<Vec<f64>> = array.outer_iter().map(|row| row.to_vec()).collect();
            data.insert(key.clone(), matrix);
        }

        serde_json::to_string_pretty(&data).map_err(|e| {
            UtilsError::InvalidParameter(format!("Failed to serialize arrays to JSON: {e}"))
        })
    }
}

// ===== DATA SERIALIZATION =====

/// Serialization utilities for ML data structures
pub struct SerializationUtils;

impl SerializationUtils {
    /// Serialize array to binary format
    pub fn serialize_array2(array: &Array2<f64>) -> UtilsResult<Vec<u8>> {
        let shape = array.shape();
        let mut data = Vec::new();

        // Write shape information
        data.extend_from_slice(&(shape[0] as u64).to_le_bytes());
        data.extend_from_slice(&(shape[1] as u64).to_le_bytes());

        // Write array data
        for &value in array.iter() {
            data.extend_from_slice(&value.to_le_bytes());
        }

        Ok(data)
    }

    /// Deserialize array from binary format
    pub fn deserialize_array2(data: &[u8]) -> UtilsResult<Array2<f64>> {
        if data.len() < 16 {
            return Err(UtilsError::InvalidParameter(
                "Insufficient data for array header".to_string(),
            ));
        }

        // Read shape information
        let nrows = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;

        let ncols = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]) as usize;

        let expected_len = 16 + nrows * ncols * 8;
        if data.len() != expected_len {
            return Err(UtilsError::InvalidParameter(format!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        // Read array data
        let mut values = Vec::with_capacity(nrows * ncols);
        for i in 0..(nrows * ncols) {
            let start = 16 + i * 8;
            let bytes = [
                data[start],
                data[start + 1],
                data[start + 2],
                data[start + 3],
                data[start + 4],
                data[start + 5],
                data[start + 6],
                data[start + 7],
            ];
            values.push(f64::from_le_bytes(bytes));
        }

        Array2::from_shape_vec((nrows, ncols), values)
            .map_err(|e| UtilsError::InvalidParameter(format!("Failed to create array: {e}")))
    }

    /// Serialize to file
    pub fn serialize_to_file<P: AsRef<Path>>(path: P, array: &Array2<f64>) -> UtilsResult<()> {
        let data = Self::serialize_array2(array)?;
        let mut writer = EfficientFileWriter::new(path, None)?;
        writer.write_data(&data)?;
        writer.flush()?;
        Ok(())
    }

    /// Deserialize from file
    pub fn deserialize_from_file<P: AsRef<Path>>(path: P) -> UtilsResult<Array2<f64>> {
        let mut reader = EfficientFileReader::new(path, None)?;
        let data = reader.read_all()?;
        Self::deserialize_array2(&data)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    #[test]
    fn test_efficient_file_reader_writer() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write test data
        let mut writer = EfficientFileWriter::new(path, None).unwrap();
        writer
            .write_lines(vec!["line1".to_string(), "line2".to_string()])
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        // Read test data
        let mut reader = EfficientFileReader::new(path, None).unwrap();
        let lines: Result<Vec<_>, _> = reader.read_lines().collect();
        let lines = lines.unwrap();

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
    }

    #[test]
    fn test_array_io() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create test array
        let original = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Write as CSV
        FormatConverter::arrays_to_csv(path, &original, None, ',').unwrap();

        // Read back as CSV
        let (header, loaded) = FormatConverter::csv_to_arrays(path, ',', false).unwrap();
        assert!(header.is_none());
        assert_eq!(original.shape(), loaded.shape());
        assert!((original - loaded).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn test_serialization() {
        let original = Array2::from_shape_vec((2, 3), vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6]).unwrap();

        // Serialize and deserialize
        let serialized = SerializationUtils::serialize_array2(&original).unwrap();
        let deserialized = SerializationUtils::deserialize_array2(&serialized).unwrap();

        assert_eq!(original.shape(), deserialized.shape());
        assert!((original - deserialized).mapv(f64::abs).sum() < 1e-10);
    }

    #[test]
    fn test_compression() {
        let data = b"Hello, World! This is a test string for compression.";

        let encoded = CompressionUtils::run_length_encode(data);
        let decoded = CompressionUtils::run_length_decode(&encoded);

        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_stream_processor() {
        let data = b"chunk1chunk2chunk3";
        let cursor = Cursor::new(data);
        let mut processor = StreamProcessor::new(cursor, 6);

        let mut chunks = Vec::new();
        processor
            .process_chunks(|chunk| {
                chunks.push(chunk.to_vec());
                Ok(())
            })
            .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(&chunks[0], b"chunk1");
        assert_eq!(&chunks[1], b"chunk2");
        assert_eq!(&chunks[2], b"chunk3");
    }

    #[test]
    fn test_format_conversion() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Create CSV content with only numeric data
        std::fs::write(path, "age,score\n25,95.5\n30,87.2").unwrap();

        let (header, data) = FormatConverter::csv_to_arrays(path, ',', true).unwrap();

        assert!(header.is_some());
        let header = header.unwrap();
        assert_eq!(header, vec!["age", "score"]);

        assert_eq!(data.shape(), &[2, 2]);
        assert!((data[[0, 0]] - 25.0).abs() < 1e-10);
        assert!((data[[0, 1]] - 95.5).abs() < 1e-10);
        assert!((data[[1, 0]] - 30.0).abs() < 1e-10);
        assert!((data[[1, 1]] - 87.2).abs() < 1e-10);
    }

    #[test]
    fn test_enhanced_json_arrays() {
        // Test JSON to arrays conversion
        let json_data = r#"
        {
            "features": [[1.0, 2.0], [3.0, 4.0]],
            "targets": [[5.0], [6.0]]
        }"#;

        let arrays = FormatConverter::json_to_arrays(json_data).unwrap();

        assert_eq!(arrays.len(), 2);
        assert!(arrays.contains_key("features"));
        assert!(arrays.contains_key("targets"));

        let features = &arrays["features"];
        assert_eq!(features.shape(), &[2, 2]);
        assert!((features[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((features[[1, 1]] - 4.0).abs() < 1e-10);

        // Test arrays to JSON conversion
        let mut array_refs = HashMap::new();
        array_refs.insert("features".to_string(), features);

        let json_result = FormatConverter::arrays_to_json(&array_refs).unwrap();
        assert!(json_result.contains("features"));
        assert!(json_result.contains("1.0"));
    }

    #[test]
    #[cfg(feature = "yaml")]
    fn test_yaml_conversion() {
        let yaml_data = r#"
        name: test
        value: 42
        nested:
          key: "hello"
        "#;

        let map = FormatConverter::yaml_to_map(yaml_data).unwrap();
        assert!(map.contains_key("name"));
        assert!(map.contains_key("value"));

        let yaml_result = FormatConverter::map_to_yaml(&map).unwrap();
        assert!(yaml_result.contains("name"));
        assert!(yaml_result.contains("42"));
    }

    #[test]
    #[cfg(feature = "toml_support")]
    fn test_toml_conversion() {
        let toml_data = r#"
        name = "test"
        value = 42

        [nested]
        key = "hello"
        "#;

        let map = FormatConverter::toml_to_map(toml_data).unwrap();
        assert!(map.contains_key("name"));
        assert!(map.contains_key("value"));

        let toml_result = FormatConverter::map_to_toml(&map).unwrap();
        assert!(toml_result.contains("name"));
        assert!(toml_result.contains("42"));
    }

    #[test]
    #[cfg(feature = "xml")]
    fn test_xml_conversion() {
        let xml_data = r#"<?xml version="1.0" encoding="UTF-8"?>
        <root>
            <name>test</name>
            <value>42</value>
        </root>"#;

        let map = FormatConverter::xml_to_simple_map(xml_data).unwrap();
        assert!(map.contains_key("name"));
        assert!(map.contains_key("value"));
        assert_eq!(map["name"], "test");
        assert_eq!(map["value"], "42");

        let xml_result = FormatConverter::simple_map_to_xml(&map, "root").unwrap();
        assert!(xml_result.contains("<name>test</name>"));
        assert!(xml_result.contains("<value>42</value>"));
    }
}
