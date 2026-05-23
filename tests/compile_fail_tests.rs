//! Compile-fail harness: drives fixtures in `tests/compile-fail/`
//! through `cargo check` and asserts expected error tokens appear in stderr.
//!
//! Each fixture is a standalone `.rs` file annotated with `//~ ERROR: <token>`
//! comments. The harness creates a temporary crate per fixture, points its
//! `[dependencies]` at the workspace root, and invokes `cargo check`.
//!
//! No external dev-dependencies are required (28-tests §4.3, §8.4).

use std::path::{Path, PathBuf};
use std::process::Command;

const WORKSPACE_REL_FROM_TMP: &str = "../../..";

struct CompileFailCase {
    name: String,
    path: PathBuf,
    expected_tokens: Vec<String>,
}

impl CompileFailCase {
    /// Parse `//~ ERROR: <token>` annotations from the fixture source.
    fn parse_expected_tokens(src: &str) -> Vec<String> {
        src.lines()
            .filter_map(|l| l.split_once("//~ ERROR:").map(|(_, rest)| rest.trim().to_string()))
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn from_path(path: PathBuf) -> Self {
        let src = std::fs::read_to_string(&path)
            .expect("compile-fail fixture must be readable");
        let name = path.file_stem().expect("compile-fail fixture must have a file stem").to_string_lossy().into_owned();
        let expected_tokens = Self::parse_expected_tokens(&src);
        Self {
            name,
            path,
            expected_tokens,
        }
    }

    /// Drive `cargo check` against a fixture using a generated temp crate
    /// that depends on `xenon` by path. This ensures `use xenon::prelude::*;`
    /// resolves.
    fn assert_compile_fail(&self) {
        let manifest_dir = std::env::current_dir()
            .expect("cwd must exist")
            .join("target/compile-fail")
            .join(&self.name);
        let src_dir = manifest_dir.join("src");
        std::fs::create_dir_all(&src_dir).expect("create temp crate dirs");

        // Copy fixture as main.rs of the temp crate.
        std::fs::copy(&self.path, src_dir.join("main.rs"))
            .expect("copy fixture into temp crate");

        // Minimal Cargo.toml referencing the workspace `xenon` crate by path.
        let cargo_toml = format!(
            r#"[package]
name = "compile_fail_{name}"
version = "0.0.0"
edition = "2024"

[[bin]]
name = "compile_fail_{name}"
path = "src/main.rs"

[dependencies]
xenon = {{ path = "{ws}" }}
"#,
            name = self.name,
            ws = WORKSPACE_REL_FROM_TMP,
        );
        std::fs::write(manifest_dir.join("Cargo.toml"), cargo_toml)
            .expect("write temp Cargo.toml");

        let output = Command::new(env!("CARGO"))
            .args(["check", "--quiet", "--manifest-path"])
            .arg(manifest_dir.join("Cargo.toml"))
            .output()
            .expect("cargo check must be invocable");

        // (1) must NOT compile.
        assert!(
            !output.status.success(),
            "fixture '{}' was expected to fail compilation but succeeded",
            self.name
        );

        // (2) stderr must contain each expected //~ ERROR: <token>.
        let stderr = String::from_utf8_lossy(&output.stderr);
        for token in &self.expected_tokens {
            assert!(
                stderr.contains(token.as_str()),
                "fixture '{}' stderr did not contain expected token {:?}\nstderr was:\n{}",
                self.name,
                token,
                stderr
            );
        }
    }
}

fn collect_compile_fail_cases(root: &Path) -> Vec<CompileFailCase> {
    std::fs::read_dir(root)
        .expect("compile-fail fixture directory must exist")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "rs"))
        .map(CompileFailCase::from_path)
        .collect()
}

// ---------------------------------------------------------------------------
// Top-level harness test (§8.4 line 1426-1432)
// ---------------------------------------------------------------------------

#[test]
fn compile_fail_harness() {
    let fixtures_dir = Path::new("tests/compile-fail");
    assert!(
        fixtures_dir.is_dir(),
        "tests/compile-fail/ must exist"
    );
    let cases = collect_compile_fail_cases(fixtures_dir);
    assert!(
        !cases.is_empty(),
        "at least one compile-fail fixture must exist"
    );
    for case in cases {
        case.assert_compile_fail();
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_compile_fail_cases_filters_rs_files() {
        let tmp = std::env::temp_dir().join("xenon_cf_collect");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).expect("temp dir creation failed");
        std::fs::write(tmp.join("a.rs"), "fn main() {}").expect("write a.rs failed");
        std::fs::write(tmp.join("not_rust.txt"), "hello").expect("write not_rust.txt failed");
        let cases = collect_compile_fail_cases(&tmp);
        assert_eq!(cases.len(), 1);
        assert!(cases
            .iter()
            .all(|c| c.path.extension().is_some_and(|e| e == "rs")));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn test_error_token_matches() {
        let src = "use xenon::prelude::*; //~ ERROR: NotADimension does not implement Dimension\nfn main(){}";
        let tokens = CompileFailCase::parse_expected_tokens(src);
        assert_eq!(
            tokens,
            vec!["NotADimension does not implement Dimension".to_string()]
        );
    }

    #[test]
    fn test_compile_fail_harness_discovery() {
        let fixtures_dir = std::path::Path::new("tests/compile-fail");
        if !fixtures_dir.is_dir() {
            return;
        }
        let cases = collect_compile_fail_cases(fixtures_dir);
        assert!(
            !cases.is_empty(),
            "compile-fail fixtures directory must contain .rs files"
        );
    }
}