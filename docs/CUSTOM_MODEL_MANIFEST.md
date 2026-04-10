# Custom Model Manifest

Place custom manifest files in:

`~/.local/share/ave/models/downloaded/*.avemodel`

The manifest format is plain text `key=value` pairs.

Example:

```ini
id=my_clearreality_x4
display_name=My ClearReality x4
family_id=clearreality-v1
family_name=ClearReality V1
stage=upscale
capabilities=denoise,deblur,upscale
fused=true
selective=false
source_format=onnx
precision=fp16
scale=4
file=my_clearreality_x4.onnx
control.enable_denoise=denoise.enabled
control.restore_strength=denoise.strength
control.upscale_width=upscale.width
description=Single-pass restoration + upscale custom export.
```

Supported keys:

- `id`: unique model id
- `display_name`: UI label
- `family_id`: stable family key used for grouping/fusion
- `family_name`: UI family name
- `stage`: primary stage for this model
- `capabilities`: comma-separated stage list
- `fused`: `true` if one pass can satisfy multiple capabilities
- `selective`: `true` if arbitrary subsets of `capabilities` are valid
- `source_format`: `onnx`, `ncnn`, or `pytorch`
- `precision`: `fp32`, `fp16`, or `int8`
- `scale`: integer upscale factor
- `fps_mul`: interpolation multiplier hint
- `file`: primary model filename or absolute path
- `file_aux`: NCNN auxiliary filename or absolute path
- `description`: UI description
- `download_url`: optional URL for later download support
- `download_url_aux`: optional auxiliary URL
- `default`: marks the model as the preferred default within its stage
- `min_vram_mib`: minimum VRAM hint
- `control.<input_name>`: explicit scalar auxiliary-input binding for custom fused MiGraphX models

Behavior notes:

- If `capabilities` contains more than one stage and `fused=true`, the runtime will try to collapse consecutive matching requests into one AI execution group.
- If `selective=false`, fusion only happens when the requested stages match the full declared capability set.
- If `selective=true`, the runtime can fuse any requested subset of the declared capability set in one pass.
- `file` and `file_aux` may be absolute paths, or filenames relative to the `downloaded/` directory.
- When a family is fused, every requested stage knob is copied into the combined request under `fused.<stage_kind>.<param>`. Example: `fused.denoise.strength=0.8`.
- The combined request also carries `fused.<stage_kind>.enabled` for every declared capability. Requested capabilities are `true`; omitted capabilities are `false`.
- Custom MiGraphX models can auto-bind scalar auxiliary inputs by naming them after a capability flag or a scoped parameter. Examples:
  - `enable_denoise`
  - `upscale_width`
  - `stereo_3d_divergence`
  - `color_fix_gamma`
- Explicit manifest bindings take precedence over the automatic naming heuristic.
- Binding expressions supported by `control.<input_name>`:
  - `<stage>`: capability enable flag, for example `denoise`
  - `<stage>.enabled`: explicit enable flag, for example `upscale.enabled`
  - `<stage>.<param>`: scoped stage knob, for example `stereo_3d.divergence`
  - `<param>`: raw primary-stage parameter key
  - `literal:<number|bool>`: fixed scalar value, for example `literal:1` or `literal:true`
