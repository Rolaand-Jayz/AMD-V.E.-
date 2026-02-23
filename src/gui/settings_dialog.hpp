#pragma once

#include <QDialog>

#include "ave/app_settings.hpp"

class QCheckBox;
class QComboBox;
class QDialogButtonBox;
class QLineEdit;
class QSpinBox;
class QTabWidget;

// ─────────────────────────────────────────────────────────────────
// SettingsDialog
// ─────────────────────────────────────────────────────────────────
// Modal dialog for editing AppSettings.  Opens in a separate window
// like ModelManagerDialog.  Changes are applied via the OK / Apply
// buttons and persisted to ~/.config/ave/settings.ini.
//
// Sections:
//   Inference   — global quantization default, default backend
//   Compilation — GPU tuning, exhaustive search
//   Encode      — codec, CRF, preset, thread count
//   Interface   — verbose log
// ─────────────────────────────────────────────────────────────────
class SettingsDialog final : public QDialog {
    Q_OBJECT

  public:
    explicit SettingsDialog(ave::AppSettings& settings,
                            QWidget* parent = nullptr);
    ~SettingsDialog() override = default;

  private slots:
    void onApplyClicked();

  private:
    void buildUi();
    void loadFromSettings();   // populate widgets from settings_
    void applyToSettings();    // write widgets back to settings_ and save

    ave::AppSettings& settings_;

    // ── Inference tab ─────────────────────────────────────────
    QComboBox* globalQuantCombo_ = nullptr;
    QComboBox* defaultBackendCombo_ = nullptr;

    // ── Compilation tab ───────────────────────────────────────
    QCheckBox* gpuTuningCheck_    = nullptr;
    QCheckBox* exhaustiveCheck_   = nullptr;

    // ── Encode tab ────────────────────────────────────────────
    QLineEdit* codecEdit_         = nullptr;
    QSpinBox*  crfSpin_           = nullptr;
    QLineEdit* presetEdit_        = nullptr;
    QSpinBox*  threadsSpin_       = nullptr;

    // ── Interface tab ─────────────────────────────────────────
    QCheckBox* verboseLogCheck_   = nullptr;
};
