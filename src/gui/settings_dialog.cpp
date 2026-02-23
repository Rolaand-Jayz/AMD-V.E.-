#include "settings_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>

#include "ave/app_settings.hpp"

using namespace ave;

// ─────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────

SettingsDialog::SettingsDialog(AppSettings& settings, QWidget* parent)
    : QDialog(parent), settings_(settings) {
    setWindowTitle(tr("Settings"));
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    setMinimumWidth(520);
    buildUi();
    loadFromSettings();
}

// ─────────────────────────────────────────────────────────────────
// UI construction
// ─────────────────────────────────────────────────────────────────

void SettingsDialog::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(8);

    auto* tabs = new QTabWidget;

    // ═══════════════════════════════════════════════════════════
    // Tab 1 — Inference
    // ═══════════════════════════════════════════════════════════
    {
        auto* tab    = new QWidget;
        auto* layout = new QVBoxLayout(tab);
        layout->setContentsMargins(12, 12, 12, 12);
        layout->setSpacing(12);

        auto* quantGroup  = new QGroupBox(tr("Quantization"));
        auto* quantLayout = new QFormLayout(quantGroup);
        quantLayout->setRowWrapPolicy(QFormLayout::WrapLongRows);

        globalQuantCombo_ = new QComboBox;
        globalQuantCombo_->addItem(tr("fp32  (full precision, safest)"),   QStringLiteral("fp32"));
        globalQuantCombo_->addItem(tr("fp16  (half precision, recommended)"), QStringLiteral("fp16"));
        globalQuantCombo_->addItem(tr("int8  (8-bit, fastest, check output quality)"), QStringLiteral("int8"));
        quantLayout->addRow(tr("Global default:"), globalQuantCombo_);

        auto* quantNote = new QLabel(
            tr("<small><i>fp16 and int8 require MiGraphX (ROCm). "
               "If MiGraphX is unavailable the pipeline falls back to fp32 automatically. "
               "Per-model overrides (set in Model Manager) take priority over this setting.</i></small>"));
        quantNote->setWordWrap(true);
        quantLayout->addRow(quantNote);
        layout->addWidget(quantGroup);

        auto* backendGroup  = new QGroupBox(tr("Backend"));
        auto* backendLayout = new QFormLayout(backendGroup);
        backendLayout->setRowWrapPolicy(QFormLayout::WrapLongRows);

        defaultBackendCombo_ = new QComboBox;
        defaultBackendCombo_->addItem(tr("Auto  (MiGraphX → NCNN-Vulkan → FFmpeg)"), QStringLiteral("auto"));
        defaultBackendCombo_->addItem(tr("MiGraphX  (ROCm GPU inference)"),           QStringLiteral("migraphx"));
        defaultBackendCombo_->addItem(tr("NCNN-Vulkan  (Vulkan GPU inference)"),      QStringLiteral("ncnn-vulkan"));
        backendLayout->addRow(tr("Default backend:"), defaultBackendCombo_);
        layout->addWidget(backendGroup);

        layout->addStretch();
        tabs->addTab(tab, tr("Inference"));
    }

    // ═══════════════════════════════════════════════════════════
    // Tab 2 — Compilation
    // ═══════════════════════════════════════════════════════════
    {
        auto* tab    = new QWidget;
        auto* layout = new QVBoxLayout(tab);
        layout->setContentsMargins(12, 12, 12, 12);
        layout->setSpacing(12);

        auto* compileGroup  = new QGroupBox(tr("MiGraphX Compilation"));
        auto* compileLayout = new QVBoxLayout(compileGroup);

        gpuTuningCheck_ = new QCheckBox(
            tr("Enable GPU tuning by default  (passes --gpu to migraphx-driver)"));
        exhaustiveCheck_ = new QCheckBox(
            tr("Enable exhaustive kernel search  (very slow; more optimised kernels)"));
        compileLayout->addWidget(gpuTuningCheck_);
        compileLayout->addWidget(exhaustiveCheck_);

        auto* compileNote = new QLabel(
            tr("<small><i>GPU tuning produces faster inference kernels but the first compile "
               "per model may take 30+ minutes. These settings apply when you click "
               "\"Convert to MiGraphX\" or \"Optimise for Hardware\" in the Model Manager.</i></small>"));
        compileNote->setWordWrap(true);
        compileLayout->addWidget(compileNote);
        layout->addWidget(compileGroup);

        layout->addStretch();
        tabs->addTab(tab, tr("Compilation"));
    }

    // ═══════════════════════════════════════════════════════════
    // Tab 3 — Encode
    // ═══════════════════════════════════════════════════════════
    {
        auto* tab    = new QWidget;
        auto* layout = new QVBoxLayout(tab);
        layout->setContentsMargins(12, 12, 12, 12);
        layout->setSpacing(12);

        auto* encGroup  = new QGroupBox(tr("FFmpeg Encode Defaults"));
        auto* encLayout = new QFormLayout(encGroup);
        encLayout->setRowWrapPolicy(QFormLayout::WrapLongRows);

        codecEdit_ = new QLineEdit;
        codecEdit_->setPlaceholderText("libx264");
        encLayout->addRow(tr("Codec:"), codecEdit_);

        crfSpin_ = new QSpinBox;
        crfSpin_->setRange(0, 51);
        crfSpin_->setToolTip(tr("Constant Rate Factor. Lower = higher quality and larger file size."));
        encLayout->addRow(tr("CRF:"), crfSpin_);

        presetEdit_ = new QLineEdit;
        presetEdit_->setPlaceholderText("medium");
        encLayout->addRow(tr("Preset:"), presetEdit_);

        threadsSpin_ = new QSpinBox;
        threadsSpin_->setRange(0, 64);
        threadsSpin_->setSpecialValueText(tr("0 (auto)"));
        threadsSpin_->setToolTip(tr("0 lets FFmpeg decide based on CPU core count."));
        encLayout->addRow(tr("FFmpeg threads:"), threadsSpin_);

        layout->addWidget(encGroup);
        layout->addStretch();
        tabs->addTab(tab, tr("Encode"));
    }

    // ═══════════════════════════════════════════════════════════
    // Tab 4 — Interface
    // ═══════════════════════════════════════════════════════════
    {
        auto* tab    = new QWidget;
        auto* layout = new QVBoxLayout(tab);
        layout->setContentsMargins(12, 12, 12, 12);
        layout->setSpacing(12);

        auto* uiGroup  = new QGroupBox(tr("Interface"));
        auto* uiLayout = new QVBoxLayout(uiGroup);

        verboseLogCheck_ = new QCheckBox(tr("Verbose session log  (show model contracts, env vars, interop status)"));
        uiLayout->addWidget(verboseLogCheck_);
        layout->addWidget(uiGroup);

        layout->addStretch();
        tabs->addTab(tab, tr("Interface"));
    }

    root->addWidget(tabs);

    // ── Button row ───────────────────────────────────────────────
    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Apply | QDialogButtonBox::Cancel);
    root->addWidget(buttons);

    connect(buttons->button(QDialogButtonBox::Ok), &QPushButton::clicked,
            this, [this]() { applyToSettings(); accept(); });
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked,
            this, &SettingsDialog::onApplyClicked);
    connect(buttons->button(QDialogButtonBox::Cancel), &QPushButton::clicked,
            this, &QDialog::reject);
}

// ─────────────────────────────────────────────────────────────────
// Populate widgets from current settings_
// ─────────────────────────────────────────────────────────────────

void SettingsDialog::loadFromSettings() {
    // Global quantization
    const QString precStr = QString::fromStdString(toString(settings_.globalQuantization));
    for (int i = 0; i < globalQuantCombo_->count(); ++i) {
        if (globalQuantCombo_->itemData(i).toString() == precStr) {
            globalQuantCombo_->setCurrentIndex(i);
            break;
        }
    }

    // Default backend
    const QString backStr = QString::fromStdString(toString(settings_.defaultBackend));
    for (int i = 0; i < defaultBackendCombo_->count(); ++i) {
        if (defaultBackendCombo_->itemData(i).toString() == backStr) {
            defaultBackendCombo_->setCurrentIndex(i);
            break;
        }
    }

    gpuTuningCheck_->setChecked(settings_.gpuTuningByDefault);
    exhaustiveCheck_->setChecked(settings_.exhaustiveTuning);

    codecEdit_->setText(QString::fromStdString(settings_.defaultCodec));
    crfSpin_->setValue(settings_.defaultCrf);
    presetEdit_->setText(QString::fromStdString(settings_.defaultPreset));
    threadsSpin_->setValue(settings_.ffmpegThreads);

    verboseLogCheck_->setChecked(settings_.verboseLog);
}

// ─────────────────────────────────────────────────────────────────
// Write widget values back to settings_ and persist
// ─────────────────────────────────────────────────────────────────

void SettingsDialog::applyToSettings() {
    // Quantization
    if (auto opt = precisionFromString(
            globalQuantCombo_->currentData().toString().toStdString())) {
        settings_.globalQuantization = *opt;
    }

    // Backend
    if (auto opt = backendTypeFromString(
            defaultBackendCombo_->currentData().toString().toStdString())) {
        settings_.defaultBackend = *opt;
    }

    settings_.gpuTuningByDefault = gpuTuningCheck_->isChecked();
    settings_.exhaustiveTuning   = exhaustiveCheck_->isChecked();

    const QString codec  = codecEdit_->text().trimmed();
    settings_.defaultCodec   = codec.isEmpty()  ? "libx264" : codec.toStdString();
    settings_.defaultCrf     = crfSpin_->value();
    const QString preset = presetEdit_->text().trimmed();
    settings_.defaultPreset  = preset.isEmpty() ? "medium"  : preset.toStdString();
    settings_.ffmpegThreads  = threadsSpin_->value();

    settings_.verboseLog = verboseLogCheck_->isChecked();

    settings_.save();
}

void SettingsDialog::onApplyClicked() {
    applyToSettings();
    // Reload in case save adjusted anything
    loadFromSettings();
}
