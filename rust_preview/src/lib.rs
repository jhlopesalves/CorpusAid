use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, Read};
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

#[pymodule]
fn corpus_preview(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_preview, m)?)?;
    m.add_function(wrap_pyfunction!(scan_directory, m)?)?;
    Ok(())
}
