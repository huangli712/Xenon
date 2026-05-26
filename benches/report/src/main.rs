//! Benchmark regression reporter.
//!
//! Parses benchmark stdout, compares against a baseline JSON,
//! and flags regressions based on configurable thresholds.
//!
//! ## Usage
//!
//! ```text
//! bench-report --input bench-out.txt --baseline baseline.json [--output report.json]
//! ```
//!
//! ## Regression Bands (27-benchmark §6.4)
//!
//! | delta      | band    | effect          |
//! |------------|---------|-----------------|
//! | > 20%      | failure | exit code 1     |
//! | > 5%       | warning | stderr warning  |
//! | ≤ 5%       | noise   | ignored         |

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::process;

use clap::Parser;
use regex::Regex;
use serde::Serialize;

// ── CLI ──────────────────────────────────────────────────────────────

/// Benchmark regression reporter.
#[derive(Parser)]
#[command(version, about)]
struct Args {
    /// Benchmark stdout text file (criterion output)
    #[arg(short, long)]
    input: PathBuf,

    /// Baseline JSON file
    #[arg(short, long)]
    baseline: PathBuf,

    /// Write JSON report (default: stdout-only summary)
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ── Parsing ──────────────────────────────────────────────────────────

/// Matches three benchmark output formats produced by xenon benches:
///   1. Plain:               `elem_add_f64/65536: 12345 ns`
///   2. 2D shape:            `sum_2d_axis0/256x256: 12345 ns`
///   3. SIMD/parallel path:  `simd_add_compare_f32/65536/simd: 12345 ns`
///                           `par_sum_compare_i64/16777216/parallel: 12345 ns`
fn bench_regex() -> &'static Regex {
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(
            r"^(?P<name>\w+)/(?P<size>\d+(?:x\d+)?)(?:/(?P<path>scalar|simd|serial|parallel))?: (?P<ns>\d+) ns",
        )
        .expect("benchmark line regex must compile")
    });
    &RE
}

fn parse_bench_output(path: &PathBuf) -> BTreeMap<String, u64> {
    let mut results = BTreeMap::new();
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read input file {}: {e}", path.display());
        process::exit(2);
    });

    let re = bench_regex();
    for line in text.lines() {
        let Some(caps) = re.captures(line) else {
            continue;
        };
        let name = caps.name("name").unwrap().as_str();
        let size = caps.name("size").unwrap().as_str();
        let path_suffix = caps.name("path").map(|m| m.as_str());
        let ns: u64 = caps.name("ns").unwrap().as_str().parse().unwrap_or(0);

        let key = match path_suffix {
            Some(p) => format!("{name}/{size}/{p}"),
            None => format!("{name}/{size}"),
        };
        results.insert(key, ns);
    }
    results
}

fn load_baseline(path: &PathBuf) -> BTreeMap<String, u64> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read baseline file {}: {e}", path.display());
        process::exit(2);
    });
    serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("ERROR: invalid baseline JSON: {e}");
        process::exit(2);
    })
}

// ── Classification ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Band {
    Failure,
    Warning,
    Noise,
}

impl Band {
    /// Classify delta against 27-benchmark §6.4 thresholds.
    fn classify(delta: f64) -> Self {
        if delta > 0.20 {
            Band::Failure
        } else if delta > 0.05 {
            Band::Warning
        } else {
            Band::Noise
        }
    }
}

// ── Report ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ReportEntry {
    current_ns: u64,
    baseline_ns: u64,
    delta: f64,
    band: Band,
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args = Args::parse();

    let current = parse_bench_output(&args.input);
    let baseline = load_baseline(&args.baseline);

    let mut report: BTreeMap<String, ReportEntry> = BTreeMap::new();
    let mut all_ok = true;

    for (key, cur_ns) in &current {
        let Some(&base_ns) = baseline.get(key) else {
            eprintln!("WARNING: no baseline for {key}");
            continue;
        };
        let delta = (*cur_ns as f64 - base_ns as f64) / base_ns as f64;
        let band = Band::classify(delta);

        let pct = delta * 100.0;

        if band == Band::Failure {
            eprintln!("REGRESSION: {key} ({pct:+.1}%)");
            all_ok = false;
        } else if band == Band::Warning {
            eprintln!("WARNING: {key} ({pct:+.1}%)");
        }

        report.insert(
            key.clone(),
            ReportEntry {
                current_ns: *cur_ns,
                baseline_ns: base_ns,
                delta,
                band,
            },
        );
    }

    if let Some(output) = &args.output {
        let json = serde_json::to_string_pretty(&report).unwrap_or_else(|e| {
            eprintln!("ERROR: failed to serialize report: {e}");
            process::exit(2);
        });
        fs::write(output, json).unwrap_or_else(|e| {
            eprintln!("ERROR: cannot write report to {}: {e}", output.display());
            process::exit(2);
        });
    }

    if !all_ok {
        process::exit(1);
    }
}
