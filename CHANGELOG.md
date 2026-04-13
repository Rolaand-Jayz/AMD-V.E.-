# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- Installed release documentation for licensing, security, release operations, and packaging.
- Native package checksum sidecars and release checksum manifest generation.
- A benchmark snapshot workflow and documentation surfacing for current MiGraphX results.

### Changed

- Package metadata now declares the MIT license consistently across Arch and RPM packaging templates.
- Release workflow documentation now reflects checksum publication and packaged release documentation.
- User-facing repository documentation has been rewritten to separate support expectations, packaging details, and benchmark evidence more clearly.

### Security

- Enforced HTTPS-only model downloads with explicit TLS verification, redirect limits, and network timeouts.
- Hardened temporary files and directories used by FFmpeg stderr capture, archive extraction, ONNX rewrite helpers, and PyTorch export helpers.
- Rejected unsafe shell-bound FFmpeg codec, preset, and profile tokens before spawning FFmpeg.
- Validated model download filenames so downloaded artifacts cannot escape the model cache directory.
- Cleaned up partial download artifacts when multi-file or archive-backed model downloads fail.
- Fixed model download bookkeeping to use the live model record, including custom manifest entries, instead of assuming every downloadable model is part of the built-in catalog.
