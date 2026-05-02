use lambda_azure_kernel::bulk_format::{create_mock_bulk, BulkHeader};
use lambda_azure_kernel::{BulkSpace, PQRouter, TernaryKernel};
use std::env;
use std::path::{Path, PathBuf};

struct Config {
    bulk_path: PathBuf,
    expert_count: u32,
    expert_size_mb: u32,
}

impl Config {
    fn from_env() -> Self {
        let mut config = Self {
            bulk_path: PathBuf::from("lae_bulk.bin"),
            expert_count: 4,
            expert_size_mb: 1,
        };

        if let Ok(path) = env::var("LAE_BULK_FILE") {
            config.bulk_path = PathBuf::from(path);
        }
        if let Ok(value) = env::var("LAE_EXPERTS") {
            if let Ok(parsed) = value.parse() {
                config.expert_count = parsed;
            }
        }
        if let Ok(value) = env::var("LAE_EXPERT_MB") {
            if let Ok(parsed) = value.parse() {
                config.expert_size_mb = parsed;
            }
        }
        config
    }
}

fn main() -> anyhow::Result<()> {
    println!("--- LAMBDA AZURE ENGINE: OMEGA POINT v4.1 (HARDENED) ---");
    let config = Config::from_env();

    if !Path::new(&config.bulk_path).exists() {
        println!(
            "[1/4] Creating compact bulk file at {:?} ({} experts × {} MB)",
            config.bulk_path, config.expert_count, config.expert_size_mb
        );
        let header = BulkHeader::new(
            config.expert_count,
            config.expert_size_mb * 1024 * 1024,
        );
        create_mock_bulk(&config.bulk_path, header)?;
    }

    let bulk = BulkSpace::open(&config.bulk_path)?;
    println!(
        "[1/4] BulkSpace ready ({} experts, {} MB each).",
        bulk.expert_count(),
        config.expert_size_mb
    );

    let router = PQRouter {
        centroids: vec![0.0; 64 * 256],
    };
    let input_embed = vec![42.0; 4096];
    let expert_id = router.select_expert(&input_embed) % bulk.expert_count();
    println!("[2/4] PQ Router expert selection: Expert ID #{} active.", expert_id);

    let expert_data = bulk.prefetch_expert(expert_id)?;
    println!(
        "[3/4] Expert Prefetch: {} bytes loaded to DMA buffer.",
        expert_data.len()
    );

    let word_count = (expert_data.len() / 8).min(128);
    let x_vec = vec![0x5555555555555555u64; word_count];
    let w_row =
        unsafe { std::slice::from_raw_parts(expert_data.as_ptr() as *const u64, word_count) };
    let result = TernaryKernel::batch_dot(w_row, x_vec.as_slice());
    println!("[4/4] Wavefront compute complete. XOR-Sum result: {}", result);

    println!("--- LAE-v4: END-TO-END DEMO COMPLETE. ---");
    Ok(())
}
