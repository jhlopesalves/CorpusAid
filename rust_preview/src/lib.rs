use chardetng::EncodingDetector;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read};
use unicode_segmentation::UnicodeSegmentation;
use walkdir::WalkDir;

fn read_preview(path: &str, limit: usize) -> Result<(String, bool), io::Error> {
    let mut file = File::open(path)?;

    if limit == 0 {
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        return Ok((buffer, false));
    }

    let mut buf = vec![0u8; limit + 1];
    let mut total_read = 0usize;
    while total_read < buf.len() {
        match file.read(&mut buf[total_read..]) {
            Ok(0) => break,
            Ok(n) => total_read += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    buf.truncate(total_read);

    let truncated = buf.len() > limit;
    if truncated {
        buf.truncate(limit);
    }

    let text = String::from_utf8_lossy(&buf).to_string();
    Ok((text, truncated))
}

fn decode_with_chardet(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }

    let mut detector = EncodingDetector::new();
    detector.feed(bytes, true);
    let encoding = detector.guess(None, true);
    let (text, _, had_errors) = encoding.decode(bytes);
    if had_errors {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        text.into_owned()
    }
}

fn count_words(text: &str, use_simple_split: bool) -> u64 {
    if use_simple_split {
        text.split_whitespace().filter(|segment| !segment.is_empty()).count() as u64
    } else {
        UnicodeSegmentation::unicode_words(text).count() as u64
    }
}

#[pyfunction]
fn load_preview(path: &str, limit: usize) -> PyResult<(String, bool)> {
    if path.is_empty() {
        return Err(PyValueError::new_err("Path cannot be empty"));
    }

    read_preview(path, limit).map_err(|err| PyIOError::new_err(err.to_string()))
}

#[pyfunction]
fn scan_directory(path: &str) -> PyResult<Vec<String>> {
    if path.is_empty() {
        return Err(PyValueError::new_err("Directory path cannot be empty"));
    }

    let walker = WalkDir::new(path).into_iter();
    let mut results = Vec::new();

    for entry in walker {
        let entry = entry.map_err(|err| PyIOError::new_err(err.to_string()))?;
        if entry.file_type().is_file() {
            let path = entry.path();
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext.eq_ignore_ascii_case("txt") {
                    results.push(path.to_string_lossy().into_owned());
                }
            }
        }
    }

    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (paths, use_processed, word_tokenization, processed_lookup=None, progress_callback=None))]
fn generate_report_summary(
    py: Python<'_>,
    paths: Vec<String>,
    use_processed: bool,
    word_tokenization: bool,
    processed_lookup: Option<HashMap<String, String>>,
    progress_callback: Option<PyObject>,
) -> PyResult<Py<PyDict>> {
    let total_files = paths.len();
    let mut total_bytes: u64 = 0;
    let mut total_words: u64 = 0;

    let processed_lookup = processed_lookup.unwrap_or_default();
    let progress_callback = progress_callback.as_ref();

    if total_files == 0 {
        if let Some(callback) = progress_callback {
            callback.call1(py, (100,))?;
        }
    }

    for (idx, path) in paths.iter().enumerate() {
        let (text, bytes): (Cow<'_, str>, u64) = if use_processed {
            if let Some(processed_text) = processed_lookup.get(path) {
                (
                    Cow::Borrowed(processed_text.as_str()),
                    processed_text.as_bytes().len() as u64,
                )
            } else {
                let data = fs::read(path)
                    .map_err(|err| PyIOError::new_err(err.to_string()))?;
                let decoded = decode_with_chardet(&data);
                (Cow::Owned(decoded), data.len() as u64)
            }
        } else {
            let data = fs::read(path)
                .map_err(|err| PyIOError::new_err(err.to_string()))?;
            let decoded = decode_with_chardet(&data);
            (Cow::Owned(decoded), data.len() as u64)
        };

        total_bytes += bytes;
        total_words += count_words(&text, word_tokenization);

        if let Some(callback) = progress_callback {
            let percent = if total_files == 0 {
                100
            } else {
                ((idx + 1) * 100 / total_files) as i32
            };
            callback.call1(py, (percent,))?;
        }
    }

    let dict = PyDict::new_bound(py);
    dict.set_item("total_files", total_files)?;
    let total_size_mb = total_bytes as f64 / (1024.0 * 1024.0);
    dict.set_item("total_size", total_size_mb)?;
    let avg_size_mb = if total_files > 0 {
        total_size_mb / total_files as f64
    } else {
        0.0
    };
    dict.set_item("avg_size", avg_size_mb)?;
    dict.set_item("total_words", total_words)?;
    let avg_words = if total_files > 0 {
        total_words as f64 / total_files as f64
    } else {
        0.0
    };
    dict.set_item("avg_words", avg_words)?;

    Ok(dict.into())
}


#[pyfunction]
fn load_full_text(path: &str) -> PyResult<String> {
    if path.is_empty() {
        return Err(PyValueError::new_err("Path cannot be empty"));
    }

    let data = fs::read(path).map_err(|err| PyIOError::new_err(err.to_string()))?;
    Ok(decode_with_chardet(&data))
}

#[pyfunction]
fn load_full_texts(paths: Vec<String>) -> PyResult<Vec<(String, String)>> {
    let mut results = Vec::with_capacity(paths.len());

    for path in paths {
        if path.is_empty() {
            return Err(PyValueError::new_err("Path cannot be empty"));
        }

        let data = fs::read(&path).map_err(|err| PyIOError::new_err(err.to_string()))?;
        let text = decode_with_chardet(&data);
        results.push((path, text));
    }

    Ok(results)
}



#[pymodule]
fn corpus_preview(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_preview, m)?)?;
    m.add_function(wrap_pyfunction!(scan_directory, m)?)?;
    m.add_function(wrap_pyfunction!(generate_report_summary, m)?)?;
    m.add_function(wrap_pyfunction!(load_full_text, m)?)?;
    m.add_function(wrap_pyfunction!(load_full_texts, m)?)?;
    Ok(())
}
