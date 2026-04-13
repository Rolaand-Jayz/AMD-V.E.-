# GUI Guide

This folder contains the optional Qt desktop frontend for AMD Video Enhancer. The GUI is not a separate product; it is another way of driving the same core pipeline that the CLI uses.

## What lives here

| File | Purpose |
| --- | --- |
| `main_gui.cpp` | GUI entrypoint that starts the Qt application |
| `main_window.cpp` / `main_window.hpp` | main application window, top-level workflow wiring, and user-facing actions |
| `filter_browser.cpp` / `filter_browser.hpp` | browsing and selecting filters/stages in a user-friendly way |
| `model_manager_dialog.cpp` / `model_manager_dialog.hpp` | model download/manage UI |
| `settings_dialog.cpp` / `settings_dialog.hpp` | persisted app settings, preferences, and runtime options |
| `toggle_switch.cpp` / `toggle_switch.hpp` | custom UI control used by the desktop frontend |

## How to think about the GUI

The GUI is a translator layer between people and the core pipeline.

It helps users:

- choose stages
- manage models
- tweak settings
- preview what they are about to do
- queue work without memorizing CLI syntax

But the heavy lifting still happens in the shared C++ core, not inside the widgets themselves.

## Best reading order

1. `main_gui.cpp`
2. `main_window.cpp`
3. `model_manager_dialog.cpp`
4. `settings_dialog.cpp`
5. `filter_browser.cpp`
6. `toggle_switch.cpp`

That order goes from app startup to the main workflow to the supporting UI pieces.

## Good companion docs

- source-tree overview: [`../README.md`](../README.md)
- public interfaces: [`../../include/ave/README.md`](../../include/ave/README.md)
- technical architecture: [`../../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md`](../../docs/GOLD_STANDARD_FOR_IMPLEMENTATION.md)
