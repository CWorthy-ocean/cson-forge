# Machine configuration

C-SON Forge uses a configuration system to manage paths and system-specific settings.

## System Detection

The system is automatically detected based on the hostname and platform. Supported systems:

- `mac` - macOS systems
- `anvil` - Anvil HPC system
- `perlmutter` - Perlmutter HPC system
- `unknown` - Fallback for other systems

## Data Paths

Data paths are automatically configured based on the detected system:

- **Source data** (`config.paths.source_data`): External datasets (GLORYS, UNIFIED_BGC, SRTM15, etc.)
- **Input data** (`config.paths.input_data`): Generated ROMS-MARBL input files
- **Run directory** (`config.paths.run_dir`): Model execution directories
- **Code root** (`config.paths.code_root`): Location of ROMS and MARBL source code repositories

### Inspecting Configuration

You can inspect the detected system and configured paths using the `config.py` CLI:

```bash
python workflows/config.py show-paths
```

This will display:
- The detected system tag (e.g., `mac`, `anvil`, `perlmutter`)
- The hostname
- All configured data paths (source_data, input_data, run_dir, code_root, blueprints, etc.)

To output the paths in JSON format:

```bash
python workflows/config.py show-paths --json
```

## Customization

To customize paths or add a new system, edit `workflows/config.py` and add a new system layout function.



