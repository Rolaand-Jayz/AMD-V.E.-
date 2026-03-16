#include "settings_dialog.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>
#include <QStringList>

#include "ave/app_settings.hpp"
#include "ave/backend_manager.hpp"
#include "ave/runtime_paths.hpp"

using namespace ave;

namespace {

QString htmlEscape(const QString& value) {
    return value.toHtmlEscaped();
}

QString availabilityLine(const QString& name, const bool available, const QString& detail) {
    const QString state = available ? "available" : "unavailable";
    const QString color = available ? "#1f6f43" : "#8b2d2d";
    return "<b>" + htmlEscape(name) + "</b>: <span style=\"color:" + color + ";\">"
        + state + "</span> <span style=\"color:#666;\">(" + htmlEscape(detail) + ")</span>";
}

QString toolLine(const QString& name, const QString& path) {
    if (path.isEmpty()) {
        return "<b>" + htmlEscape(name) + "</b>: <span style=\"color:#8b2d2d;\">missing</span>";
    }
    return "<b>" + htmlEscape(name) + "</b>: <span style=\"color:#1f6f43;\">"
        + htmlEscape(path) + "</span>";
}

QString modelsRootPath() {
    return QString::fromStdString(defaultWritableModelsDir().string());
}

QWidget* wrapPageHeader(const QString& title,
                        const QString& subtitle,
                        QWidget* parent,
                        QVBoxLayout** outLayout) {
    auto* page = new QWidget(parent);
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(12);

    auto* titleLabel = new QLabel(title, page);
    titleLabel->setStyleSheet("font-size: 18px; font-weight: 700;");
    layout->addWidget(titleLabel);

    auto* subtitleLabel = new QLabel(subtitle, page);
    subtitleLabel->setWordWrap(true);
    subtitleLabel->setStyleSheet("color: #666;");
    layout->addWidget(subtitleLabel);

    *outLayout = layout;
    return page;
}

}  // namespace

SettingsDialog::SettingsDialog(AppSettings& settings, QWidget* parent)
    : QDialog(parent), settings_(settings) {
    setWindowTitle(tr("Workspace Settings"));
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    setMinimumSize(940, 680);
    buildUi();
    loadFromSettings();
}

QString SettingsDialog::configDirectoryPath() {
    return QFileInfo(QString::fromStdString(AppSettings::defaultPath())).dir().absolutePath();
}

QString SettingsDialog::modelsDirectoryPath() const {
    return modelsRootPath();
}

void SettingsDialog::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(14, 14, 14, 14);
    root->setSpacing(12);

    auto* hero = new QFrame(this);
    hero->setStyleSheet(
        "QFrame { background: #f4efe4; border: 1px solid #d8cfbd; border-radius: 12px; }"
        "QLabel { background: transparent; }");
    auto* heroLayout = new QVBoxLayout(hero);
    heroLayout->setContentsMargins(16, 14, 16, 14);
    heroLayout->setSpacing(6);

    auto* heroTitle = new QLabel("Workspace Settings", hero);
    heroTitle->setStyleSheet("font-size: 22px; font-weight: 700; color: #2f2416;");
    heroLayout->addWidget(heroTitle);

    auto* heroBody = new QLabel(
        "Set sane defaults for jobs, previews, output naming, and session behavior. "
        "Diagnostics on the left show whether the local toolchain is actually ready.",
        hero);
    heroBody->setWordWrap(true);
    heroBody->setStyleSheet("color: #5d5140;");
    heroLayout->addWidget(heroBody);
    root->addWidget(hero);

    auto* contentRow = new QHBoxLayout;
    contentRow->setContentsMargins(0, 0, 0, 0);
    contentRow->setSpacing(12);

    navList_ = new QListWidget(this);
    navList_->addItems({
        tr("Overview"),
        tr("Workflow"),
        tr("Processing"),
        tr("Encode"),
        tr("Interface")
    });
    navList_->setFixedWidth(180);
    navList_->setCurrentRow(0);
    navList_->setStyleSheet(
        "QListWidget { border: 1px solid #d7d7d7; border-radius: 10px; padding: 6px; }"
        "QListWidget::item { padding: 10px 12px; border-radius: 8px; }"
        "QListWidget::item:selected { background: #e8efe7; color: #1c3b2a; font-weight: 600; }");
    contentRow->addWidget(navList_);

    pageStack_ = new QStackedWidget(this);
    pageStack_->setStyleSheet(
        "QGroupBox { font-weight: 600; }"
        "QLineEdit, QComboBox, QSpinBox { min-height: 28px; }");

    {
        QVBoxLayout* layout = nullptr;
        auto* page = wrapPageHeader(
            tr("System Overview"),
            tr("See where settings are stored, where models live, and whether the local backend/tooling stack is usable."),
            pageStack_, &layout);

        auto* pathsGroup = new QGroupBox(tr("Workspace Paths"), page);
        auto* pathsLayout = new QFormLayout(pathsGroup);
        settingsPathLabel_ = new QLabel(pathsGroup);
        settingsPathLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        settingsPathLabel_->setWordWrap(true);
        pathsLayout->addRow(tr("Settings file:"), settingsPathLabel_);
        modelsPathLabel_ = new QLabel(pathsGroup);
        modelsPathLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        modelsPathLabel_->setWordWrap(true);
        pathsLayout->addRow(tr("Model store:"), modelsPathLabel_);
        rememberedPathsLabel_ = new QLabel(pathsGroup);
        rememberedPathsLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        rememberedPathsLabel_->setWordWrap(true);
        pathsLayout->addRow(tr("Remembered paths:"), rememberedPathsLabel_);
        layout->addWidget(pathsGroup);

        auto* backendGroup = new QGroupBox(tr("Backend Readiness"), page);
        auto* backendLayout = new QVBoxLayout(backendGroup);
        backendSummaryLabel_ = new QLabel(backendGroup);
        backendSummaryLabel_->setWordWrap(true);
        backendSummaryLabel_->setTextFormat(Qt::RichText);
        backendSummaryLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        backendLayout->addWidget(backendSummaryLabel_);
        layout->addWidget(backendGroup);

        auto* toolGroup = new QGroupBox(tr("Toolchain Snapshot"), page);
        auto* toolLayout = new QVBoxLayout(toolGroup);
        toolchainSummaryLabel_ = new QLabel(toolGroup);
        toolchainSummaryLabel_->setWordWrap(true);
        toolchainSummaryLabel_->setTextFormat(Qt::RichText);
        toolchainSummaryLabel_->setTextInteractionFlags(Qt::TextSelectableByMouse);
        toolLayout->addWidget(toolchainSummaryLabel_);
        layout->addWidget(toolGroup);

        auto* actionRow = new QHBoxLayout;
        auto* configBtn = new QPushButton(tr("Open Config Folder"), page);
        auto* modelsBtn = new QPushButton(tr("Open Model Store"), page);
        auto* reloadBtn = new QPushButton(tr("Refresh Diagnostics"), page);
        actionRow->addWidget(configBtn);
        actionRow->addWidget(modelsBtn);
        actionRow->addWidget(reloadBtn);
        actionRow->addStretch();
        layout->addLayout(actionRow);

        connect(configBtn, &QPushButton::clicked, this, &SettingsDialog::onOpenConfigDirectory);
        connect(modelsBtn, &QPushButton::clicked, this, &SettingsDialog::onOpenModelsDirectory);
        connect(reloadBtn, &QPushButton::clicked, this, &SettingsDialog::onReloadDiagnostics);

        layout->addStretch();
        pageStack_->addWidget(page);
    }

    {
        QVBoxLayout* layout = nullptr;
        auto* page = wrapPageHeader(
            tr("Workflow Defaults"),
            tr("These defaults shape how the main workspace behaves before a job is run."),
            pageStack_, &layout);

        auto* defaultsGroup = new QGroupBox(tr("Run Defaults"), page);
        auto* defaultsLayout = new QFormLayout(defaultsGroup);
        defaultDryRunCheck_ = new QCheckBox(tr("Start new jobs in dry-run mode"), defaultsGroup);
        defaultsLayout->addRow(QString(), defaultDryRunCheck_);
        previewDurationSpin_ = new QSpinBox(defaultsGroup);
        previewDurationSpin_->setRange(1, 300);
        previewDurationSpin_->setSuffix(" sec");
        defaultsLayout->addRow(tr("Preview duration:"), previewDurationSpin_);
        previewFrameIntervalSpin_ = new QSpinBox(defaultsGroup);
        previewFrameIntervalSpin_->setRange(1, 240);
        previewFrameIntervalSpin_->setSuffix(" frames");
        previewFrameIntervalSpin_->setToolTip(
            tr("Lower values update the live preview more often but add overhead."));
        defaultsLayout->addRow(tr("Preview frame interval:"), previewFrameIntervalSpin_);
        layout->addWidget(defaultsGroup);

        auto* outputGroup = new QGroupBox(tr("Output Naming"), page);
        auto* outputLayout = new QFormLayout(outputGroup);
        autoFillOutputCheck_ = new QCheckBox(
            tr("Auto-fill the output path when an input video is chosen"),
            outputGroup);
        outputLayout->addRow(QString(), autoFillOutputCheck_);
        outputSuffixEdit_ = new QLineEdit(outputGroup);
        outputSuffixEdit_->setPlaceholderText("_enhanced");
        outputLayout->addRow(tr("Auto suffix:"), outputSuffixEdit_);
        rememberPathsCheck_ = new QCheckBox(
            tr("Restore the last input and output paths on startup"),
            outputGroup);
        outputLayout->addRow(QString(), rememberPathsCheck_);
        layout->addWidget(outputGroup);

        auto* guardrailGroup = new QGroupBox(tr("Guardrails"), page);
        auto* guardrailLayout = new QVBoxLayout(guardrailGroup);
        confirmRunCheck_ = new QCheckBox(tr("Ask for confirmation before starting a run"), guardrailGroup);
        confirmClearPipelineCheck_ = new QCheckBox(
            tr("Ask for confirmation before clearing the pipeline"),
            guardrailGroup);
        guardrailLayout->addWidget(confirmRunCheck_);
        guardrailLayout->addWidget(confirmClearPipelineCheck_);
        layout->addWidget(guardrailGroup);

        layout->addStretch();
        pageStack_->addWidget(page);
    }

    {
        QVBoxLayout* layout = nullptr;
        auto* page = wrapPageHeader(
            tr("Processing Defaults"),
            tr("Set the backend strategy the workspace should prefer when new jobs are assembled."),
            pageStack_, &layout);

        auto* backendGroup = new QGroupBox(tr("Backend Preference"), page);
        auto* backendLayout = new QFormLayout(backendGroup);
        defaultBackendCombo_ = new QComboBox(backendGroup);
        defaultBackendCombo_->addItem(tr("Auto (MiGraphX -> NCNN -> Vulkan Compute)"), QStringLiteral("auto"));
        defaultBackendCombo_->addItem(tr("MiGraphX (ROCm inference)"), QStringLiteral("migraphx"));
        defaultBackendCombo_->addItem(tr("NCNN Vulkan"), QStringLiteral("ncnn-vulkan"));
        defaultBackendCombo_->addItem(tr("Vulkan Compute"), QStringLiteral("vulkan"));
        defaultBackendCombo_->addItem(tr("VapourSynth"), QStringLiteral("vapoursynth"));
        defaultBackendCombo_->addItem(tr("GLSL Shader"), QStringLiteral("glsl"));
        backendLayout->addRow(tr("Default backend:"), defaultBackendCombo_);
        layout->addWidget(backendGroup);

        auto* note = new QLabel(
            tr("This only sets the default selection in the workspace. Actual availability still depends on the local machine, runtime libraries, and installed plugins."),
            page);
        note->setWordWrap(true);
        note->setStyleSheet("color: #666;");
        layout->addWidget(note);
        layout->addStretch();
        pageStack_->addWidget(page);
    }

    {
        QVBoxLayout* layout = nullptr;
        auto* page = wrapPageHeader(
            tr("Encode Defaults"),
            tr("Choose the delivery encode the GUI should default to when a new job is prepared."),
            pageStack_, &layout);

        auto* encodeGroup = new QGroupBox(tr("FFmpeg Defaults"), page);
        auto* encodeLayout = new QFormLayout(encodeGroup);

        codecCombo_ = new QComboBox(encodeGroup);
        codecCombo_->setEditable(true);
        codecCombo_->addItems({
            "libx264", "libx265", "libsvtav1",
            "h264_vaapi", "hevc_vaapi", "av1_vaapi",
            "h264_amf", "hevc_amf", "av1_amf",
            "libvpx-vp9", "libaom-av1", "ffv1", "utvideo"
        });
        encodeLayout->addRow(tr("Codec:"), codecCombo_);

        profileEdit_ = new QLineEdit(encodeGroup);
        profileEdit_->setPlaceholderText("(auto)");
        encodeLayout->addRow(tr("Profile:"), profileEdit_);

        crfSpin_ = new QSpinBox(encodeGroup);
        crfSpin_->setRange(0, 63);
        encodeLayout->addRow(tr("CRF:"), crfSpin_);

        presetEdit_ = new QLineEdit(encodeGroup);
        presetEdit_->setPlaceholderText("medium");
        encodeLayout->addRow(tr("Preset:"), presetEdit_);

        threadsSpin_ = new QSpinBox(encodeGroup);
        threadsSpin_->setRange(0, 128);
        threadsSpin_->setSpecialValueText(tr("0 (auto)"));
        encodeLayout->addRow(tr("FFmpeg threads:"), threadsSpin_);

        layout->addWidget(encodeGroup);

        auto* encodeHint = new QLabel(
            tr("Thread count now feeds the actual FFmpeg encode commands used by the GUI. Leave it at 0 to let FFmpeg pick based on the system."),
            page);
        encodeHint->setWordWrap(true);
        encodeHint->setStyleSheet("color: #666;");
        layout->addWidget(encodeHint);

        layout->addStretch();
        pageStack_->addWidget(page);
    }

    {
        QVBoxLayout* layout = nullptr;
        auto* page = wrapPageHeader(
            tr("Interface"),
            tr("Tune how the session log and workspace feel while jobs are running."),
            pageStack_, &layout);

        auto* uiGroup = new QGroupBox(tr("Session View"), page);
        auto* uiLayout = new QVBoxLayout(uiGroup);
        autoScrollLogCheck_ = new QCheckBox(tr("Auto-scroll the session log as new lines arrive"), uiGroup);
        verboseLogCheck_ = new QCheckBox(
            tr("Verbose session log (show extra backend and environment detail when available)"),
            uiGroup);
        uiLayout->addWidget(autoScrollLogCheck_);
        uiLayout->addWidget(verboseLogCheck_);
        layout->addWidget(uiGroup);

        auto* uiHint = new QLabel(
            tr("Verbose logging depends on how much detail the active backend emits. The workspace will still suppress duplicate progress spam."),
            page);
        uiHint->setWordWrap(true);
        uiHint->setStyleSheet("color: #666;");
        layout->addWidget(uiHint);

        layout->addStretch();
        pageStack_->addWidget(page);
    }

    contentRow->addWidget(pageStack_, 1);
    root->addLayout(contentRow, 1);

    auto* footerRow = new QHBoxLayout;
    auto* resetBtn = new QPushButton(tr("Reset to Defaults"), this);
    auto* reloadBtn = new QPushButton(tr("Reload Diagnostics"), this);
    footerRow->addWidget(resetBtn);
    footerRow->addWidget(reloadBtn);
    footerRow->addStretch();

    auto* buttons = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Apply | QDialogButtonBox::Cancel,
        this);
    footerRow->addWidget(buttons);
    root->addLayout(footerRow);

    connect(navList_, &QListWidget::currentRowChanged, pageStack_, &QStackedWidget::setCurrentIndex);
    connect(resetBtn, &QPushButton::clicked, this, &SettingsDialog::onResetDefaults);
    connect(reloadBtn, &QPushButton::clicked, this, &SettingsDialog::onReloadDiagnostics);
    connect(rememberPathsCheck_, &QCheckBox::toggled, this, [this](bool) { refreshDiagnostics(); });
    connect(buttons->button(QDialogButtonBox::Ok), &QPushButton::clicked,
            this, [this]() { applyToSettings(); accept(); });
    connect(buttons->button(QDialogButtonBox::Apply), &QPushButton::clicked,
            this, &SettingsDialog::onApplyClicked);
    connect(buttons->button(QDialogButtonBox::Cancel), &QPushButton::clicked,
            this, &QDialog::reject);
}

void SettingsDialog::loadFromSettings() {
    loadWidgetsFromSettings(settings_);
    refreshDiagnostics();
}

void SettingsDialog::loadWidgetsFromSettings(const AppSettings& settings) {
    const QString backend = QString::fromStdString(toString(settings.defaultBackend));
    for (int i = 0; i < defaultBackendCombo_->count(); ++i) {
        if (defaultBackendCombo_->itemData(i).toString() == backend) {
            defaultBackendCombo_->setCurrentIndex(i);
            break;
        }
    }

    codecCombo_->setCurrentText(QString::fromStdString(settings.defaultCodec));
    profileEdit_->setText(QString::fromStdString(settings.defaultProfile));
    crfSpin_->setValue(settings.defaultCrf);
    presetEdit_->setText(QString::fromStdString(settings.defaultPreset));
    threadsSpin_->setValue(settings.ffmpegThreads);

    defaultDryRunCheck_->setChecked(settings.defaultDryRun);
    previewDurationSpin_->setValue(settings.defaultPreviewDurationSec);
    previewFrameIntervalSpin_->setValue(settings.previewFrameInterval);
    autoFillOutputCheck_->setChecked(settings.autoFillOutputPath);
    outputSuffixEdit_->setText(QString::fromStdString(settings.outputSuffix));
    rememberPathsCheck_->setChecked(settings.rememberLastPaths);
    confirmRunCheck_->setChecked(settings.confirmBeforeRun);
    confirmClearPipelineCheck_->setChecked(settings.confirmBeforeClearPipeline);

    autoScrollLogCheck_->setChecked(settings.autoScrollLog);
    verboseLogCheck_->setChecked(settings.verboseLog);
}

void SettingsDialog::applyToSettings() {
    if (auto opt = backendTypeFromString(
            defaultBackendCombo_->currentData().toString().toStdString())) {
        settings_.defaultBackend = *opt;
    }

    const QString codec = codecCombo_->currentText().trimmed();
    settings_.defaultCodec = codec.isEmpty() ? "libx264" : codec.toStdString();
    settings_.defaultProfile = profileEdit_->text().trimmed().toStdString();
    settings_.defaultCrf = crfSpin_->value();
    const QString preset = presetEdit_->text().trimmed();
    settings_.defaultPreset = preset.isEmpty() ? "medium" : preset.toStdString();
    settings_.ffmpegThreads = threadsSpin_->value();

    settings_.defaultDryRun = defaultDryRunCheck_->isChecked();
    settings_.defaultPreviewDurationSec = previewDurationSpin_->value();
    settings_.previewFrameInterval = previewFrameIntervalSpin_->value();
    settings_.autoFillOutputPath = autoFillOutputCheck_->isChecked();
    const QString suffix = outputSuffixEdit_->text().trimmed();
    settings_.outputSuffix = suffix.isEmpty() ? "_enhanced" : suffix.toStdString();
    settings_.rememberLastPaths = rememberPathsCheck_->isChecked();
    settings_.confirmBeforeRun = confirmRunCheck_->isChecked();
    settings_.confirmBeforeClearPipeline = confirmClearPipelineCheck_->isChecked();

    settings_.autoScrollLog = autoScrollLogCheck_->isChecked();
    settings_.verboseLog = verboseLogCheck_->isChecked();

    if (!settings_.rememberLastPaths) {
        settings_.lastInputPath.clear();
        settings_.lastOutputPath.clear();
    }

    settings_.save();
    refreshDiagnostics();
    emit settingsApplied();
}

void SettingsDialog::refreshDiagnostics() {
    settingsPathLabel_->setText(QString::fromStdString(AppSettings::defaultPath()));
    modelsPathLabel_->setText(modelsDirectoryPath());

    QString remembered;
    if (!rememberPathsCheck_->isChecked()) {
        remembered = tr("Path restore is disabled.");
    } else {
        const QString in = QString::fromStdString(settings_.lastInputPath).trimmed();
        const QString out = QString::fromStdString(settings_.lastOutputPath).trimmed();
        remembered = tr("Input: %1\nOutput: %2")
            .arg(in.isEmpty() ? tr("(none)") : in,
                 out.isEmpty() ? tr("(none)") : out);
    }
    rememberedPathsLabel_->setText(remembered);

    BackendManager manager;
    const auto infos = manager.probeBackends();
    QStringList backendLines;
    for (const auto& info : infos) {
        backendLines << availabilityLine(
            QString::fromStdString(info.name),
            info.available,
            QString::fromStdString(info.detail));
    }
    backendSummaryLabel_->setText(backendLines.join("<br/>"));

    const QString ffmpeg = QStandardPaths::findExecutable("ffmpeg");
    const QString ffprobe = QStandardPaths::findExecutable("ffprobe");
    const QString vspipe = QStandardPaths::findExecutable("vspipe");
    const QString mpv = QStandardPaths::findExecutable("mpv");
    const QString glslc = QStandardPaths::findExecutable("glslc");
    const QString rocminfo = QStandardPaths::findExecutable("rocminfo");

    toolchainSummaryLabel_->setText(QStringList{
        toolLine("ffmpeg", ffmpeg),
        toolLine("ffprobe", ffprobe),
        toolLine("vspipe", vspipe),
        toolLine("mpv", mpv),
        toolLine("glslc", glslc),
        toolLine("rocminfo", rocminfo)
    }.join("<br/>"));
}

void SettingsDialog::onApplyClicked() {
    applyToSettings();
    loadFromSettings();
}

void SettingsDialog::onResetDefaults() {
    if (QMessageBox::question(
            this,
            tr("Reset Settings"),
            tr("Reset the dialog fields to built-in defaults? This does not save until you click Apply or OK."))
            != QMessageBox::Yes) {
        return;
    }
    loadWidgetsFromSettings(AppSettings{});
    refreshDiagnostics();
}

void SettingsDialog::onReloadDiagnostics() {
    refreshDiagnostics();
}

void SettingsDialog::onOpenConfigDirectory() {
    const QString path = configDirectoryPath();
    QDir().mkpath(path);
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void SettingsDialog::onOpenModelsDirectory() {
    const QString path = modelsDirectoryPath();
    QDir().mkpath(path);
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}
