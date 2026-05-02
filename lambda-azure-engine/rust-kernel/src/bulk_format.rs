use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{anyhow, Context, Result};

pub const BULK_MAGIC: [u8; 4] = *b"LAEB";
pub const BULK_VERSION: u8 = 1;
pub const BULK_HEADER_LEN: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct BulkHeader {
    pub version: u8,
    pub expert_count: u32,
    pub expert_size_bytes: u32,
}

impl BulkHeader {
    pub fn new(expert_count: u32, expert_size_bytes: u32) -> Self {
        Self {
            version: BULK_VERSION,
            expert_count,
            expert_size_bytes,
        }
    }

    pub fn to_bytes(self) -> [u8; BULK_HEADER_LEN] {
        let mut buf = [0u8; BULK_HEADER_LEN];
        buf[..4].copy_from_slice(&BULK_MAGIC);
        buf[4] = self.version;
        buf[8..12].copy_from_slice(&self.expert_count.to_le_bytes());
        buf[12..16].copy_from_slice(&self.expert_size_bytes.to_le_bytes());
        buf
    }

    pub fn from_bytes(buf: [u8; BULK_HEADER_LEN]) -> Result<Self> {
        if buf[..4] != BULK_MAGIC {
            return Err(anyhow!("Invalid bulk magic header"));
        }
        Ok(Self {
            version: buf[4],
            expert_count: u32::from_le_bytes(buf[8..12].try_into()?),
            expert_size_bytes: u32::from_le_bytes(buf[12..16].try_into()?),
        })
    }
}

pub fn read_bulk_header(path: &Path) -> Result<Option<BulkHeader>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let mut buf = [0u8; BULK_HEADER_LEN];
    let bytes = file.read(&mut buf)?;
    if bytes < BULK_HEADER_LEN || buf[..4] != BULK_MAGIC {
        return Ok(None);
    }
    Ok(Some(BulkHeader::from_bytes(buf)?))
}

pub fn create_mock_bulk(path: &Path, header: BulkHeader) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("create bulk file {:?}", path))?;
    let total_len = BULK_HEADER_LEN as u64
        + header.expert_count as u64 * header.expert_size_bytes as u64;
    file.set_len(total_len)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header.to_bytes())?;
    Ok(())
}
