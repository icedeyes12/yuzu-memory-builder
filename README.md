# Yuzu Memory Builder

Rebuild episodic & semantic memories from 7k+ chat history using local DuckDB + ONNX embeddings, then batch migrate to Supabase.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SUPABASE      │────▶│  MEMORY BUILDER  │────▶│   SUPABASE      │
│  (Export Data)  │     │  (DuckDB+ONNX)   │     │  (Batch Insert) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        ▼                       ▼                        ▼
   Read-only              Local Process            Safe Insert
   SELECT *              Embedding + FSRS         ON CONFLICT
```

## Quick Start

```bash
# 1. Clone di Codespace (16GB RAM)
git clone https://github.com/YOUR_USERNAME/yuzu-memory-builder.git
cd yuzu-memory-builder

# 2. Setup (auto-install dependencies)
make setup

# 3. Config environment
cp .env.example .env
# Edit .env dengan Supabase credentials

# 4. Run TUI
python main.py
```

## Requirements

- **Codespace**: 16GB RAM (Large tier)
- **Python**: 3.12+
- **Storage**: ~2GB free (models + DuckDB)

## CLI Commands

| Command | Description |
|---------|-------------|
| `export` | Export Supabase → DuckDB |
| `embed` | Generate ONNX embeddings |
| `validate` | Spot-check quality |
| `preview` | Preview memories |
| `migrate` | Batch insert to Supabase |
| `status` | Show progress |

## Safety Guarantees

- ✅ Read-only dari Supabase (SELECT only)
- ✅ All writes go to local DuckDB first
- ✅ Validation before any production insert
- ✅ Dry-run mode available
- ✅ Rollback checkpoints

## License

MIT
