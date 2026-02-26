#include "main_window.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <QApplication>
#include <QClipboard>
#include <QComboBox>
#include <QDateTime>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QMimeData>
#include <QMetaObject>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QStringList>
#include <QTextCursor>
#include <QUrl>
#include <QVBoxLayout>
#include <QWidget>

#include "ave/backend.hpp"
#include "ave/model_catalog.hpp"
#include "ave/stage.hpp"
#include "ave/types.hpp"
#include "ave/video_processor.hpp"
#include "model_manager_dialog.hpp"
#include "settings_dialog.hpp"
#include "toggle_switch.hpp"

// ─────────────────────────────────────────────────────────────────
// Anonymous namespace helpers
// ─────────────────────────────────────────────────────────────────
namespace {

QString toQString(const std::string& v) { return QString::fromStdString(v); }

QString stageTitle(ave::StageKind kind) {
    switch (kind) {
        case ave::StageKind::RestoreCompression: return "Restore Compression";
        case ave::StageKind::RemoveArtifacts:    return "Remove Artifacts";
        case ave::StageKind::Denoise:            return "Denoise";
        case ave::StageKind::Deblur:             return "Deblur";
        case ave::StageKind::Dehalo:             return "Dehalo";
        case ave::StageKind::ColorFix:           return "Color Fix";
        case ave::StageKind::Upscale:            return "Upscale";
        case ave::StageKind::Sharpen:            return "Sharpen";
        case ave::StageKind::Interpolate:        return "Interpolate";
    }
    return "Unknown";
}

// Slider range: 0–200, value / 100.0 = 0.00–2.00
QString sliderPct(int v) {
    return QString::number(v / 100.0, 'f', 2);
}

bool parseInt64(const QString& s, std::int64_t& out) {
    bool ok = false;
    qlonglong v = s.toLongLong(&ok);
    if (!ok) return false;
    out = static_cast<std::int64_t>(v);
    return true;
}

bool parseDouble(const QString& s, double& out) {
    bool ok = false;
    double v = s.toDouble(&ok);
    if (!ok) return false;
    out = v;
    return true;
}

ave::ParameterValue parseParameterValue(const QString& text) {
    const QString t = text.trimmed();
    if (t.compare("true",  Qt::CaseInsensitive) == 0) return true;
    if (t.compare("false", Qt::CaseInsensitive) == 0) return false;
    std::int64_t iv = 0;
    if (parseInt64(t, iv)) return iv;
    double dv = 0.0;
    if (parseDouble(t, dv)) return dv;
    return t.toStdString();
}

QJsonObject parameterToJson(const ave::ParameterValue& v) {
    QJsonObject o;
    if (std::holds_alternative<std::int64_t>(v)) {
        o.insert("type", "int");
        o.insert("value", QString::number(std::get<std::int64_t>(v)));
        return o;
    }
    if (std::holds_alternative<double>(v)) {
        o.insert("type", "double");
        o.insert("value", std::get<double>(v));
        return o;
    }
    if (std::holds_alternative<bool>(v)) {
        o.insert("type", "bool");
        o.insert("value", std::get<bool>(v));
        return o;
    }
    o.insert("type", "string");
    o.insert("value", QString::fromStdString(std::get<std::string>(v)));
    return o;
}

std::optional<ave::ParameterValue> parameterFromJson(const QJsonObject& o) {
    const QString type = o.value("type").toString();
    if (type == "int") {
        std::int64_t iv = 0;
        if (!parseInt64(o.value("value").toString(), iv)) return std::nullopt;
        return iv;
    }
    if (type == "double") return o.value("value").toDouble();
    if (type == "bool")   return o.value("value").toBool();
    if (type == "string") return o.value("value").toString().toStdString();
    return std::nullopt;
}

std::optional<ave::BackendType> backendFromString(const QString& s) {
    const QString n = s.trimmed().toLower();
    if (n == "auto")    return ave::BackendType::Auto;
    if (n == "migraphx") return ave::BackendType::MiGraphX;
    if (n == "ncnn-vulkan" || n == "ncnn") return ave::BackendType::NcnnVulkan;
    return std::nullopt;
}

QString quoteArg(const QString& s) {
    QString out = "\"";
    for (QChar ch : s) {
        if (ch == '\\' || ch == '"') out.append('\\');
        out.append(ch);
    }
    out.append('"');
    return out;
}

std::vector<ave::EnhancementStage> quickTemplateStages(int index) {
    std::vector<ave::EnhancementStage> out;
    if (index == 1) {
        ave::EnhancementStage r; r.kind = ave::StageKind::RestoreCompression; r.params["strength"] = 0.9;
        ave::EnhancementStage a; a.kind = ave::StageKind::RemoveArtifacts;    a.params["strength"] = 0.8;
        ave::EnhancementStage i; i.kind = ave::StageKind::Interpolate;        i.params["fps"] = std::int64_t{60};
        out = {r, a, i};
    } else if (index == 2) {
        ave::EnhancementStage d; d.kind = ave::StageKind::Denoise;      d.params["strength"] = 0.55;
        ave::EnhancementStage u; u.kind = ave::StageKind::Upscale;      u.params["width"] = std::int64_t{3840}; u.params["height"] = std::int64_t{2160};
        ave::EnhancementStage s; s.kind = ave::StageKind::Sharpen;      s.params["amount"] = 0.45;
        ave::EnhancementStage i; i.kind = ave::StageKind::Interpolate;  i.params["fps"] = std::int64_t{60};
        out = {d, u, s, i};
    } else if (index == 3) {
        ave::EnhancementStage r; r.kind = ave::StageKind::RestoreCompression; r.params["strength"] = 1.0;
        ave::EnhancementStage b; b.kind = ave::StageKind::Deblur;             b.params["strength"] = 0.65;
        ave::EnhancementStage u; u.kind = ave::StageKind::Upscale;            u.params["width"] = std::int64_t{2560}; u.params["height"] = std::int64_t{1440};
        out = {r, b, u};
    }
    return out;
}

// Index in paramStack_ for each StageKind
// Order matches the stacked widget page order
int paramPageForKind(ave::StageKind k) {
    switch (k) {
        case ave::StageKind::RestoreCompression:
        case ave::StageKind::RemoveArtifacts:
        case ave::StageKind::Denoise:
        case ave::StageKind::Deblur:
        case ave::StageKind::Dehalo:    return 0; // strength
        case ave::StageKind::ColorFix:  return 1; // colorfix sliders
        case ave::StageKind::Upscale:   return 2;
        case ave::StageKind::Sharpen:   return 3;
        case ave::StageKind::Interpolate: return 4;
    }
    return 5; // empty
}

} // namespace

// ─────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("AMD Video Enhancer");
    resize(1320, 900);
    setAcceptDrops(true);

    appSettings_.load();
    buildUi();
    wireActions();

    modelManager_.refresh();
    refreshModelCombo();
    refreshParamPanel();
    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
}

// ─────────────────────────────────────────────────────────────────
// Parameter panel builders
// ─────────────────────────────────────────────────────────────────

QWidget* MainWindow::buildStrengthPanel(QWidget* parent) {
    auto* w = new QWidget(parent);
    auto* lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->addWidget(new QLabel("Strength:", w));
    strengthSlider_ = new QSlider(Qt::Horizontal, w);
    strengthSlider_->setRange(0, 200);
    strengthSlider_->setValue(80);
    strengthLabel_ = new QLabel("0.80", w);
    strengthLabel_->setMinimumWidth(36);
    lay->addWidget(strengthSlider_, 1);
    lay->addWidget(strengthLabel_);
    connect(strengthSlider_, &QSlider::valueChanged, this, [this](int v) {
        strengthLabel_->setText(sliderPct(v));
        refreshCommandPreview();
    });
    return w;
}

QWidget* MainWindow::buildColorFixPanel(QWidget* parent) {
    auto* w  = new QWidget(parent);
    auto* lay = new QGridLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setHorizontalSpacing(4);

    struct Entry { const char* label; QSlider** slider; QLabel** lbl; int def; };
    const Entry entries[] = {
        {"Contrast:",   &contrastSlider_,   &contrastLabel_,   100},
        {"Brightness:", &brightnessSlider_, &brightnessLabel_, 100},
        {"Saturation:", &saturationSlider_, &saturationLabel_, 100},
        {"Gamma:",      &gammaSlider_,      &gammaLabel_,      100},
        {"Vibrance:",   &vibranceSlider_,   &vibranceLabel_,   100},
    };

    for (int r = 0; const auto& e : entries) {
        lay->addWidget(new QLabel(e.label, w), r, 0);
        *e.slider = new QSlider(Qt::Horizontal, w);
        (*e.slider)->setRange(0, 200);
        (*e.slider)->setValue(e.def);
        lay->addWidget(*e.slider, r, 1);
        *e.lbl = new QLabel(sliderPct(e.def), w);
        (*e.lbl)->setMinimumWidth(36);
        lay->addWidget(*e.lbl, r, 2);
        ++r;
    }

    auto makeConn = [this](QSlider* s, QLabel* l) {
        connect(s, &QSlider::valueChanged, this, [this, s, l](int v) {
            l->setText(sliderPct(v)); refreshCommandPreview(); });
    };
    makeConn(contrastSlider_,   contrastLabel_);
    makeConn(brightnessSlider_, brightnessLabel_);
    makeConn(saturationSlider_, saturationLabel_);
    makeConn(gammaSlider_,      gammaLabel_);
    makeConn(vibranceSlider_,   vibranceLabel_);

    return w;
}

QWidget* MainWindow::buildUpscalePanel(QWidget* parent) {
    auto* w = new QWidget(parent);
    auto* lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->addWidget(new QLabel("Width:", w));
    upscaleWidthSpin_ = new QSpinBox(w);
    upscaleWidthSpin_->setRange(16, 16384);
    upscaleWidthSpin_->setValue(3840);
    lay->addWidget(upscaleWidthSpin_);
    lay->addWidget(new QLabel("Height:", w));
    upscaleHeightSpin_ = new QSpinBox(w);
    upscaleHeightSpin_->setRange(16, 16384);
    upscaleHeightSpin_->setValue(2160);
    lay->addWidget(upscaleHeightSpin_);
    lay->addStretch();
    connect(upscaleWidthSpin_,  &QSpinBox::valueChanged, this, [this](int) { refreshCommandPreview(); });
    connect(upscaleHeightSpin_, &QSpinBox::valueChanged, this, [this](int) { refreshCommandPreview(); });
    return w;
}

QWidget* MainWindow::buildSharpenPanel(QWidget* parent) {
    auto* w = new QWidget(parent);
    auto* lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->addWidget(new QLabel("Amount:", w));
    sharpenSlider_ = new QSlider(Qt::Horizontal, w);
    sharpenSlider_->setRange(0, 200);
    sharpenSlider_->setValue(50);
    sharpenLabel_ = new QLabel("0.50", w);
    sharpenLabel_->setMinimumWidth(36);
    lay->addWidget(sharpenSlider_, 1);
    lay->addWidget(sharpenLabel_);
    connect(sharpenSlider_, &QSlider::valueChanged, this, [this](int v) {
        sharpenLabel_->setText(sliderPct(v)); refreshCommandPreview(); });
    return w;
}

QWidget* MainWindow::buildInterpolatePanel(QWidget* parent) {
    auto* w = new QWidget(parent);
    auto* lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->addWidget(new QLabel("Target FPS:", w));
    interpolateFpsSpin_ = new QSpinBox(w);
    interpolateFpsSpin_->setRange(1, 240);
    interpolateFpsSpin_->setValue(60);
    lay->addWidget(interpolateFpsSpin_);
    lay->addSpacing(12);
    lay->addWidget(new QLabel("Scene detect:", w));
    sceneDetectToggle_ = new ToggleSwitch(w);
    sceneDetectToggle_->setChecked(true);
    lay->addWidget(sceneDetectToggle_);
    lay->addStretch();
    connect(interpolateFpsSpin_, &QSpinBox::valueChanged, this, [this](int) { refreshCommandPreview(); });
    connect(sceneDetectToggle_, &ToggleSwitch::toggled,   this, [this](bool) { refreshCommandPreview(); });
    return w;
}

QWidget* MainWindow::buildEmptyPanel(QWidget* parent) {
    auto* w = new QWidget(parent);
    (void)w;
    return w;
}

// ─────────────────────────────────────────────────────────────────
// buildUi
// ─────────────────────────────────────────────────────────────────

void MainWindow::buildUi() {
    auto* central = new QWidget(this);
    setCentralWidget(central);
    auto* root = new QVBoxLayout(central);
    root->setSpacing(6);

    // ── I/O ──────────────────────────────────────────────────────
    auto* ioGroup = new QGroupBox("Input / Output", central);
    auto* ioGrid  = new QGridLayout(ioGroup);
    ioGrid->addWidget(new QLabel("Input"), 0, 0);
    inputPathEdit_ = new QLineEdit(ioGroup);
    ioGrid->addWidget(inputPathEdit_, 0, 1);
    auto* browseIn = new QPushButton("Browse…", ioGroup);
    ioGrid->addWidget(browseIn, 0, 2);
    ioGrid->addWidget(new QLabel("Output"), 1, 0);
    outputPathEdit_ = new QLineEdit(ioGroup);
    ioGrid->addWidget(outputPathEdit_, 1, 1);
    auto* browseOut = new QPushButton("Browse…", ioGroup);
    ioGrid->addWidget(browseOut, 1, 2);
    root->addWidget(ioGroup);

    connect(browseIn, &QPushButton::clicked, this, [this]() {
        const QString p = QFileDialog::getOpenFileName(this, "Select Input Video", inputPathEdit_->text());
        if (!p.isEmpty()) inputPathEdit_->setText(p);
    });
    connect(browseOut, &QPushButton::clicked, this, [this]() {
        const QString p = QFileDialog::getSaveFileName(this, "Select Output Video", outputPathEdit_->text());
        if (!p.isEmpty()) outputPathEdit_->setText(p);
    });

    // ── Encode + backend ─────────────────────────────────────────
    auto* encGroup = new QGroupBox("Backend + Encode", central);
    auto* encGrid  = new QGridLayout(encGroup);

    backendCombo_ = new QComboBox(encGroup);
    backendCombo_->addItem("Auto (MiGraphX → NCNN)", static_cast<int>(ave::BackendType::Auto));
    backendCombo_->addItem("MiGraphX (ROCm)",         static_cast<int>(ave::BackendType::MiGraphX));
    backendCombo_->addItem("NCNN (Vulkan)",            static_cast<int>(ave::BackendType::NcnnVulkan));
    codecEdit_  = new QLineEdit("libx264", encGroup);
    presetEdit_ = new QLineEdit("medium",  encGroup);
    crfSpin_    = new QSpinBox(encGroup); crfSpin_->setRange(0,51); crfSpin_->setValue(18);
    dryRunToggle_ = new ToggleSwitch(encGroup);
    auto* modelMgrBtn  = new QPushButton("Model Manager…", encGroup);
    auto* probeBtn     = new QPushButton("Probe Backends",  encGroup);
    auto* settingsBtn  = new QPushButton("Settings…",       encGroup);
    auto* saveBtn      = new QPushButton("Save Profile",    encGroup);
    auto* loadBtn      = new QPushButton("Load Profile",    encGroup);

    encGrid->addWidget(new QLabel("Backend"), 0, 0);
    encGrid->addWidget(backendCombo_, 0, 1);
    encGrid->addWidget(new QLabel("Codec"),   0, 2);
    encGrid->addWidget(codecEdit_, 0, 3);
    encGrid->addWidget(new QLabel("CRF"),     1, 0);
    encGrid->addWidget(crfSpin_, 1, 1);
    encGrid->addWidget(new QLabel("Preset"),  1, 2);
    encGrid->addWidget(presetEdit_, 1, 3);
    {
        auto* dryRow = new QHBoxLayout;
        dryRow->setContentsMargins(0,0,0,0);
        dryRow->addWidget(new QLabel("Dry Run:"));
        dryRow->addWidget(dryRunToggle_);
        dryRow->addStretch();
        encGrid->addLayout(dryRow, 2, 0, 1, 2);
    }
    encGrid->addWidget(modelMgrBtn, 2, 2);
    encGrid->addWidget(probeBtn,    2, 3);
    encGrid->addWidget(settingsBtn, 3, 0);
    encGrid->addWidget(saveBtn,     3, 2);
    encGrid->addWidget(loadBtn,     3, 3);
    root->addWidget(encGroup);

    connect(modelMgrBtn,  &QPushButton::clicked, this, &MainWindow::openModelManager);
    connect(probeBtn,     &QPushButton::clicked, this, &MainWindow::probeBackends);
    connect(settingsBtn,  &QPushButton::clicked, this, &MainWindow::openSettings);
    connect(saveBtn,      &QPushButton::clicked, this, &MainWindow::saveProfile);
    connect(loadBtn,      &QPushButton::clicked, this, &MainWindow::loadProfile);

    // ── Stage builder ─────────────────────────────────────────────
    auto* sbGroup = new QGroupBox("Stage Builder", central);
    auto* sbGrid  = new QGridLayout(sbGroup);

    stageKindCombo_ = new QComboBox(sbGroup);
    const ave::StageKind allKinds[] = {
        ave::StageKind::RestoreCompression, ave::StageKind::RemoveArtifacts,
        ave::StageKind::Denoise, ave::StageKind::Deblur, ave::StageKind::Dehalo,
        ave::StageKind::ColorFix, ave::StageKind::Upscale,
        ave::StageKind::Sharpen, ave::StageKind::Interpolate
    };
    for (auto k : allKinds)
        stageKindCombo_->addItem(stageTitle(k), static_cast<int>(k));

    modelCombo_ = new QComboBox(sbGroup);
    modelCombo_->setMinimumWidth(220);

    sbGrid->addWidget(new QLabel("Stage:"),  0, 0);
    sbGrid->addWidget(stageKindCombo_,        0, 1);
    sbGrid->addWidget(new QLabel("Model:"),   0, 2);
    sbGrid->addWidget(modelCombo_,            0, 3);

    // Parameter stacked widget (pages must match paramPageForKind())
    paramStack_ = new QStackedWidget(sbGroup);
    paramStack_->addWidget(buildStrengthPanel(sbGroup));   // 0: generic strength
    paramStack_->addWidget(buildColorFixPanel(sbGroup));   // 1: colorfix
    paramStack_->addWidget(buildUpscalePanel(sbGroup));    // 2: upscale
    paramStack_->addWidget(buildSharpenPanel(sbGroup));    // 3: sharpen
    paramStack_->addWidget(buildInterpolatePanel(sbGroup));// 4: interpolate
    paramStack_->addWidget(buildEmptyPanel(sbGroup));      // 5: empty fallback
    sbGrid->addWidget(paramStack_, 1, 0, 1, 4);

    extraParamsEdit_ = new QLineEdit(sbGroup);
    extraParamsEdit_->setPlaceholderText("extra_key=value,another=42");
    sbGrid->addWidget(new QLabel("Extra params:"), 2, 0);
    sbGrid->addWidget(extraParamsEdit_, 2, 1, 1, 3);

    auto* addStageBtn = new QPushButton("Add Stage", sbGroup);
    removeStageButton_  = new QPushButton("Remove",    sbGroup);
    moveUpButton_       = new QPushButton("▲",         sbGroup);
    moveDownButton_     = new QPushButton("▼",         sbGroup);
    clearStagesButton_  = new QPushButton("Clear All", sbGroup);

    auto* stageRow = new QHBoxLayout;
    stageRow->addWidget(addStageBtn);
    stageRow->addWidget(removeStageButton_);
    stageRow->addWidget(moveUpButton_);
    stageRow->addWidget(moveDownButton_);
    stageRow->addWidget(clearStagesButton_);
    stageRow->addStretch();
    sbGrid->addLayout(stageRow, 3, 0, 1, 4);
    root->addWidget(sbGroup);

    connect(addStageBtn,        &QPushButton::clicked, this, &MainWindow::addStage);
    connect(removeStageButton_, &QPushButton::clicked, this, &MainWindow::removeSelectedStage);
    connect(moveUpButton_,      &QPushButton::clicked, this, [this]() { moveSelectedStage(-1); });
    connect(moveDownButton_,    &QPushButton::clicked, this, [this]() { moveSelectedStage(1); });
    connect(clearStagesButton_, &QPushButton::clicked, this, &MainWindow::clearStages);

    // ── Pipeline lists ────────────────────────────────────────────
    auto* listSplit = new QSplitter(Qt::Horizontal, central);

    auto* reqGroup  = new QGroupBox("Requested Pipeline  (drag to reorder)", listSplit);
    auto* reqLayout = new QVBoxLayout(reqGroup);
    requestedStagesView_ = new QListWidget(reqGroup);
    requestedStagesView_->setDragDropMode(QAbstractItemView::InternalMove);
    requestedStagesView_->setDefaultDropAction(Qt::MoveAction);
    requestedStagesView_->setSelectionMode(QAbstractItemView::SingleSelection);
    reqLayout->addWidget(requestedStagesView_);

    auto* planGroup  = new QGroupBox("Planned Execution Order", listSplit);
    auto* planLayout = new QVBoxLayout(planGroup);
    plannedStagesView_ = new QListWidget(planGroup);
    planLayout->addWidget(plannedStagesView_);

    listSplit->addWidget(reqGroup);
    listSplit->addWidget(planGroup);
    listSplit->setStretchFactor(0, 1);
    listSplit->setStretchFactor(1, 1);
    root->addWidget(listSplit, 1);

    // ── Actions bar ───────────────────────────────────────────────
    auto* actGroup  = new QGroupBox("Run + Utilities", central);
    auto* actLayout = new QGridLayout(actGroup);

    quickTemplateCombo_ = new QComboBox(actGroup);
    quickTemplateCombo_->addItem("Quick Templates");
    quickTemplateCombo_->addItem("Web Cleanup 1080p60");
    quickTemplateCombo_->addItem("Anime Upscale 4K60");
    quickTemplateCombo_->addItem("Archive Restore 1440p");

    runButton_ = new QPushButton("▶  Run Job", actGroup);
    runButton_->setStyleSheet("QPushButton { font-weight: bold; background-color: #2196F3; color: white; padding: 4px 12px; }");
    auto* copyBtn = new QPushButton("Copy Command", actGroup);
    commandPreviewEdit_ = new QLineEdit(actGroup);
    commandPreviewEdit_->setReadOnly(true);

    progressBar_ = new QProgressBar(actGroup);
    progressBar_->setRange(0, 100);
    progressBar_->setFormat("Overall: %p%");
    progressBar_->setVisible(false);

    taskProgressBar_ = new QProgressBar(actGroup);
    taskProgressBar_->setRange(0, 100);
    taskProgressBar_->setFormat("Task: %p%");
    taskProgressBar_->setVisible(false);

    progressLabel_ = new QLabel(actGroup);
    progressLabel_->setWordWrap(true);
    progressLabel_->setStyleSheet("color: #555; font-style: italic;");
    progressLabel_->setVisible(false);

    actLayout->addWidget(quickTemplateCombo_, 0, 0);
    actLayout->addWidget(runButton_,          0, 1);
    actLayout->addWidget(copyBtn,             0, 2);
    actLayout->addWidget(new QLabel("Overall:", actGroup),     1, 0);
    actLayout->addWidget(progressBar_,        1, 1, 1, 2);
    actLayout->addWidget(new QLabel("Current task:", actGroup), 2, 0);
    actLayout->addWidget(taskProgressBar_,    2, 1, 1, 2);
    actLayout->addWidget(progressLabel_,      3, 0, 1, 3);
    actLayout->addWidget(new QLabel("Command:"), 4, 0);
    actLayout->addWidget(commandPreviewEdit_, 4, 1, 1, 2);
    root->addWidget(actGroup);

    connect(runButton_, &QPushButton::clicked, this, &MainWindow::runJob);
    connect(copyBtn, &QPushButton::clicked, this, [this]() {
        if (auto* cb = QApplication::clipboard())
            cb->setText(commandPreviewEdit_->text());
        appendLog("Command copied to clipboard.");
    });
    connect(quickTemplateCombo_, &QComboBox::currentIndexChanged, this, &MainWindow::applyQuickTemplate);

    // ── Log ───────────────────────────────────────────────────────
    auto* logGroup  = new QGroupBox("Session Log", central);
    auto* logLayout = new QVBoxLayout(logGroup);
    logView_ = new QPlainTextEdit(logGroup);
    logView_->setReadOnly(true);
    logView_->setMaximumBlockCount(2000);
    logLayout->addWidget(logView_);
    root->addWidget(logGroup, 1);
}

// ─────────────────────────────────────────────────────────────────
// wireActions
// ─────────────────────────────────────────────────────────────────

void MainWindow::wireActions() {
    auto ref = [this]() { refreshCommandPreview(); };

    connect(inputPathEdit_,  &QLineEdit::textChanged, this, [ref](const QString&) { ref(); });
    connect(outputPathEdit_, &QLineEdit::textChanged, this, [ref](const QString&) { ref(); });
    connect(codecEdit_,      &QLineEdit::textChanged, this, [ref](const QString&) { ref(); });
    connect(presetEdit_,     &QLineEdit::textChanged, this, [ref](const QString&) { ref(); });
    connect(crfSpin_,        &QSpinBox::valueChanged, this, [ref](int)           { ref(); });
    connect(backendCombo_,   &QComboBox::currentIndexChanged, this, [ref](int)   { ref(); });
    connect(dryRunToggle_,   &ToggleSwitch::toggled,  this, [ref](bool)          { ref(); });
    connect(extraParamsEdit_,&QLineEdit::textChanged, this, [ref](const QString&){ ref(); });
    connect(modelCombo_,     &QComboBox::currentIndexChanged, this, [ref](int)   { ref(); });

    connect(stageKindCombo_, &QComboBox::currentIndexChanged, this, [this](int) {
        refreshModelCombo();
        refreshParamPanel();
        refreshCommandPreview();
    });

    connect(requestedStagesView_, &QListWidget::currentRowChanged,
            this, &MainWindow::onPipelineSelectionChanged);

    // Detect drag-drop reorders
    connect(requestedStagesView_->model(), &QAbstractItemModel::rowsMoved,
            this, [this](const QModelIndex&, int src, int, const QModelIndex&, int dst) {
                // Sync internal stages_ vector to match the list's new order
                const int from = src;
                const int to   = dst > src ? dst - 1 : dst;
                if (from == to || from < 0 || to < 0 ||
                    from >= (int)stages_.size() || to >= (int)stages_.size()) return;
                const auto ufrom = static_cast<std::size_t>(from);
                const auto uto   = static_cast<std::size_t>(to);
                auto stage = stages_[ufrom];
                stages_.erase(stages_.begin() + from);
                stages_.insert(stages_.begin() + static_cast<std::ptrdiff_t>(uto), stage);
                refreshPlannedStages();
                refreshCommandPreview();
            });
}

// ─────────────────────────────────────────────────────────────────
// Model combo + param panel refresh
// ─────────────────────────────────────────────────────────────────

void MainWindow::refreshModelCombo() {
    modelCombo_->clear();
    const auto kind = static_cast<ave::StageKind>(stageKindCombo_->currentData().toInt());
    const auto entries = modelManager_.dropdownEntriesForStage(kind);
    for (const auto& e : entries)
        modelCombo_->addItem(QString::fromStdString(e.label), QString::fromStdString(e.modelId));
    // Add a "Default" option if list is empty (built-in / no-model stages)
    if (modelCombo_->count() == 0)
        modelCombo_->addItem("Default (built-in)", QString());
}

void MainWindow::refreshParamPanel() {
    const auto kind = static_cast<ave::StageKind>(stageKindCombo_->currentData().toInt());
    paramStack_->setCurrentIndex(paramPageForKind(kind));
}

// ─────────────────────────────────────────────────────────────────
// Stage list + plan refresh
// ─────────────────────────────────────────────────────────────────

void MainWindow::refreshRequestedStages() {
    const int prevRow = requestedStagesView_->currentRow();
    requestedStagesView_->clear();
    for (const auto& s : stages_)
        requestedStagesView_->addItem(stageToDisplay(s));
    if (prevRow >= 0 && prevRow < requestedStagesView_->count())
        requestedStagesView_->setCurrentRow(prevRow);
}

void MainWindow::refreshPlannedStages() {
    plannedStagesView_->clear();
    for (const auto& s : planner_.plan(stages_))
        plannedStagesView_->addItem(stageToDisplay(s));
}

void MainWindow::refreshCommandPreview() {
    QStringList args;
    args << "./build/ave";

    const QString in = inputPathEdit_->text().trimmed();
    const QString out = outputPathEdit_->text().trimmed();
    if (!in.isEmpty())  { args << "--input"  << quoteArg(in);  }
    if (!out.isEmpty()) { args << "--output" << quoteArg(out); }

    const int bd = backendCombo_->currentData().toInt();
    args << "--backend" << toQString(ave::toString(static_cast<ave::BackendType>(bd)));

    for (const auto& s : stages_)
        args << "--stage" << quoteArg(stageToCommandSpec(s));

    args << "--codec"  << codecEdit_->text().trimmed();
    args << "--crf"    << QString::number(crfSpin_->value());
    args << "--preset" << presetEdit_->text().trimmed();

    if (dryRunToggle_->isChecked()) args << "--dry-run";

    commandPreviewEdit_->setText(args.join(' '));
}

void MainWindow::onPipelineSelectionChanged() {
    const bool sel = requestedStagesView_->currentRow() >= 0;
    removeStageButton_->setEnabled(sel);
    moveUpButton_->setEnabled(sel);
    moveDownButton_->setEnabled(sel);
}

// ─────────────────────────────────────────────────────────────────
// Stage helpers
// ─────────────────────────────────────────────────────────────────

QString MainWindow::stageToDisplay(const ave::EnhancementStage& stage) {
    std::vector<std::string> keys;
    keys.reserve(stage.params.size());
    for (const auto& [k, _] : stage.params) keys.push_back(k);
    std::sort(keys.begin(), keys.end());

    QStringList parts;
    for (const auto& k : keys) {
        auto it = stage.params.find(k);
        if (it != stage.params.end())
            parts << toQString(k) + "=" + toQString(ave::parameterValueToString(it->second));
    }

    QString base = toQString(ave::toString(stage.kind));
    if (!parts.isEmpty())
        base += " (" + parts.join(", ") + ")";
    return base;
}

QString MainWindow::stageToCommandSpec(const ave::EnhancementStage& stage) {
    std::vector<std::string> keys;
    keys.reserve(stage.params.size());
    for (const auto& [k, _] : stage.params) keys.push_back(k);
    std::sort(keys.begin(), keys.end());

    QStringList pairs;
    for (const auto& k : keys) {
        auto it = stage.params.find(k);
        if (it != stage.params.end())
            pairs << toQString(k) + "=" + toQString(ave::parameterValueToString(it->second));
    }

    QString out = toQString(ave::toString(stage.kind));
    if (!pairs.isEmpty()) out += ":" + pairs.join(",");
    return out;
}

ave::EnhancementStage MainWindow::buildStageFromEditor() const {
    ave::EnhancementStage stage;
    stage.kind = static_cast<ave::StageKind>(stageKindCombo_->currentData().toInt());

    // Model
    const QString modelId = modelCombo_->currentData().toString();
    if (!modelId.isEmpty()) stage.params["model"] = modelId.toStdString();

    const int page = paramPageForKind(stage.kind);
    if (page == 0) {
        // strength panel
        stage.params["strength"] = strengthSlider_->value() / 100.0;
    } else if (page == 1) {
        // colorfix
        stage.params["contrast"]   = contrastSlider_->value()   / 100.0;
        stage.params["brightness"] = brightnessSlider_->value() / 100.0;
        stage.params["saturation"] = saturationSlider_->value() / 100.0;
        stage.params["gamma"]      = gammaSlider_->value()      / 100.0;
        stage.params["vibrance"]   = vibranceSlider_->value()   / 100.0;
    } else if (page == 2) {
        // upscale
        stage.params["width"]  = static_cast<std::int64_t>(upscaleWidthSpin_->value());
        stage.params["height"] = static_cast<std::int64_t>(upscaleHeightSpin_->value());
    } else if (page == 3) {
        // sharpen
        stage.params["amount"] = sharpenSlider_->value() / 100.0;
    } else if (page == 4) {
        // interpolate
        stage.params["fps"] = static_cast<std::int64_t>(interpolateFpsSpin_->value());
        stage.params["scene_detect"] = sceneDetectToggle_->isChecked();
    }

    // Extra params
    const QString extra = extraParamsEdit_->text().trimmed();
    if (!extra.isEmpty()) {
        for (const QString& pair : extra.split(',', Qt::SkipEmptyParts)) {
            qsizetype eq = pair.indexOf('=');
            if (eq <= 0) continue;
            const QString key = pair.left(static_cast<int>(eq)).trimmed();
            const QString val = pair.mid(static_cast<int>(eq) + 1).trimmed();
            if (!key.isEmpty()) stage.params[key.toStdString()] = parseParameterValue(val);
        }
    }
    return stage;
}

// ─────────────────────────────────────────────────────────────────
// Pipeline manipulation
// ─────────────────────────────────────────────────────────────────

void MainWindow::addStage() {
    stages_.push_back(buildStageFromEditor());
    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
}

void MainWindow::removeSelectedStage() {
    const int row = requestedStagesView_->currentRow();
    if (row < 0 || row >= (int)stages_.size()) return;
    stages_.erase(stages_.begin() + row);
    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
}

void MainWindow::moveSelectedStage(int delta) {
    const int row = requestedStagesView_->currentRow();
    if (row < 0 || row >= (int)stages_.size()) return;
    const int target = row + delta;
    if (target < 0 || target >= (int)stages_.size()) return;
    std::swap(stages_[static_cast<std::size_t>(row)], stages_[static_cast<std::size_t>(target)]);
    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
    requestedStagesView_->setCurrentRow(target);
}

void MainWindow::clearStages() {
    stages_.clear();
    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
}

// ─────────────────────────────────────────────────────────────────
// Job
// ─────────────────────────────────────────────────────────────────

std::optional<ave::VideoJob> MainWindow::buildJob(QString& error) const {
    ave::VideoJob job;
    job.inputPath       = inputPathEdit_->text().trimmed().toStdString();
    job.outputPath      = outputPathEdit_->text().trimmed().toStdString();
    job.requestedBackend= static_cast<ave::BackendType>(backendCombo_->currentData().toInt());
    job.requestedStages = stages_;
    job.encode.codec    = codecEdit_->text().trimmed().toStdString();
    job.encode.crf      = crfSpin_->value();
    job.encode.preset   = presetEdit_->text().trimmed().toStdString();
    job.dryRun          = dryRunToggle_->isChecked();

    if (!job.dryRun && job.inputPath.empty())  { error = "Input path is required.";  return std::nullopt; }
    if (!job.dryRun && job.outputPath.empty()) { error = "Output path is required."; return std::nullopt; }
    if (job.encode.codec.empty())              { error = "Codec is required.";        return std::nullopt; }
    if (job.encode.preset.empty())             { error = "Preset is required.";       return std::nullopt; }
    return job;
}

void MainWindow::setRunning(bool running) {
    isRunning_.store(running);
    runButton_->setEnabled(!running);
    progressBar_->setValue(0);
    progressBar_->setVisible(running);
    taskProgressBar_->setValue(0);
    taskProgressBar_->setVisible(running);
    progressLabel_->setVisible(running);
    if (!running) {
        progressLabel_->clear();
        lastProgressMsg_.clear();
    }
    const bool sel = requestedStagesView_->currentRow() >= 0;
    removeStageButton_->setEnabled(!running && sel);
    moveUpButton_->setEnabled(!running && sel);
    moveDownButton_->setEnabled(!running && sel);
    clearStagesButton_->setEnabled(!running);
}

void MainWindow::appendLog(const QString& line) {
    const QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    logView_->appendPlainText("[" + ts + "] " + line);
    logView_->moveCursor(QTextCursor::End);
}

void MainWindow::runJob() {
    if (isRunning_.load()) return;
    QString err;
    auto maybeJob = buildJob(err);
    if (!maybeJob) { QMessageBox::warning(this, "Invalid Configuration", err); return; }

    appendLog("Job started.");
    setRunning(true);

    ave::VideoJob job = *maybeJob;
    QPointer<MainWindow> self(this);

    // Wire up the progress callback to update both progress bars and the log.
    job.progressCb = [self](int overallPct, int taskPct, const std::string& msg) {
        const QString qmsg = QString::fromStdString(msg);
        QMetaObject::invokeMethod(self, [self, overallPct, taskPct, qmsg]() {
            if (!self) return;
            self->progressBar_->setValue(overallPct);
            self->taskProgressBar_->setValue(taskPct);
            if (!qmsg.isEmpty()) {
                self->progressLabel_->setText(qmsg);
                // Log each new milestone message (suppress duplicate per-frame updates)
                if (qmsg != self->lastProgressMsg_) {
                    self->lastProgressMsg_ = qmsg;
                    self->appendLog(qmsg);
                }
            }
        }, Qt::QueuedConnection);
    };

    std::thread([self, job]() {
        ave::VideoProcessor processor;
        std::string error;
        const bool ok = processor.process(job, error);
        if (!self) return;
        const QString resultLine = ok
            ? "Job completed successfully."
            : "Job failed: " + toQString(error);
        QMetaObject::invokeMethod(self, [self, ok, resultLine]() {
            if (!self) return;
            self->appendLog(resultLine);
            self->setRunning(false);
            if (!ok) QMessageBox::critical(self, "Processing Failed", resultLine);
        }, Qt::QueuedConnection);
    }).detach();
}

// ─────────────────────────────────────────────────────────────────
// Backends / Model Manager
// ─────────────────────────────────────────────────────────────────

void MainWindow::probeBackends() {
    ave::BackendManager manager;
    const auto infos = manager.probeBackends();
    QStringList lines;
    for (const auto& info : infos)
        lines << (toQString(info.name) + ": " +
                  (info.available ? "available" : "unavailable") +
                  " (" + toQString(info.detail) + ")");
    const QString summary = lines.join('\n');
    appendLog("Backend probe:\n" + summary);
    QMessageBox::information(this, "Backend Probe", summary);
}

void MainWindow::openModelManager() {
    ModelManagerDialog dlg(modelManager_, appSettings_, this);
    dlg.exec();
    // Refresh model dropdowns after the dialog closes (new downloads/conversions)
    refreshModelCombo();
}

void MainWindow::openSettings() {
    SettingsDialog dlg(appSettings_, this);
    dlg.exec();
    // Re-apply any backend preference changes immediately.
    refreshCommandPreview();
}

// ─────────────────────────────────────────────────────────────────
// Profiles
// ─────────────────────────────────────────────────────────────────

void MainWindow::saveProfile() {
    const QString filePath = QFileDialog::getSaveFileName(this, "Save Profile", {}, "JSON (*.json)");
    if (filePath.isEmpty()) return;

    QJsonObject root;
    root.insert("schema_version", 2);
    root.insert("input",  inputPathEdit_->text().trimmed());
    root.insert("output", outputPathEdit_->text().trimmed());
    root.insert("backend", toQString(ave::toString(
        static_cast<ave::BackendType>(backendCombo_->currentData().toInt()))));
    root.insert("codec",  codecEdit_->text().trimmed());
    root.insert("crf",    crfSpin_->value());
    root.insert("preset", presetEdit_->text().trimmed());
    root.insert("dry_run", dryRunToggle_->isChecked());

    QJsonArray stagesArr;
    for (const auto& stage : stages_) {
        QJsonObject so;
        so.insert("kind", toQString(ave::toString(stage.kind)));
        QJsonObject po;
        for (const auto& [key, val] : stage.params)
            po.insert(toQString(key), parameterToJson(val));
        so.insert("params", po);
        stagesArr.append(so);
    }
    root.insert("stages", stagesArr);

    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        QMessageBox::critical(this, "Save Failed", "Cannot write to file.");
        return;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();
    appendLog("Saved profile: " + filePath);
}

void MainWindow::loadProfile() {
    const QString filePath = QFileDialog::getOpenFileName(this, "Load Profile", {}, "JSON (*.json)");
    if (filePath.isEmpty()) return;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Load Failed", "Cannot open file.");
        return;
    }
    QJsonParseError pe;
    const auto doc = QJsonDocument::fromJson(file.readAll(), &pe);
    file.close();
    if (pe.error != QJsonParseError::NoError || !doc.isObject()) {
        QMessageBox::critical(this, "Load Failed", "Invalid JSON profile.");
        return;
    }

    const QJsonObject root = doc.object();
    inputPathEdit_->setText(root.value("input").toString());
    outputPathEdit_->setText(root.value("output").toString());

    if (auto b = backendFromString(root.value("backend").toString())) {
        int idx = backendCombo_->findData(static_cast<int>(*b));
        if (idx >= 0) backendCombo_->setCurrentIndex(idx);
    }
    codecEdit_->setText(root.value("codec").toString("libx264"));
    presetEdit_->setText(root.value("preset").toString("medium"));
    crfSpin_->setValue(root.value("crf").toInt(18));
    dryRunToggle_->setChecked(root.value("dry_run").toBool(false));

    stages_.clear();
    for (const auto& sv : root.value("stages").toArray()) {
        if (!sv.isObject()) continue;
        const QJsonObject so = sv.toObject();
        auto kind = ave::stageKindFromString(so.value("kind").toString().toStdString());
        if (!kind) continue;
        ave::EnhancementStage stage;
        stage.kind = *kind;
        const QJsonObject po = so.value("params").toObject();
        for (auto it = po.begin(); it != po.end(); ++it) {
            if (!it.value().isObject()) continue;
            auto pv = parameterFromJson(it.value().toObject());
            if (pv) stage.params[it.key().toStdString()] = *pv;
        }
        stages_.push_back(stage);
    }

    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
    appendLog("Loaded profile: " + filePath);
}

void MainWindow::applyQuickTemplate(int index) {
    if (index <= 0) return;
    stages_ = quickTemplateStages(index);
    refreshRequestedStages();
    refreshPlannedStages();
    refreshCommandPreview();
    if (index == 1) outputPathEdit_->setText(QFileInfo(inputPathEdit_->text()).path() + "/output/" + QFileInfo(inputPathEdit_->text()).completeBaseName() + "_cleanup_60fps.mp4");
    if (index == 2) outputPathEdit_->setText(QFileInfo(inputPathEdit_->text()).path() + "/output/" + QFileInfo(inputPathEdit_->text()).completeBaseName() + "_anime_4k_60fps.mp4");
    if (index == 3) outputPathEdit_->setText(QFileInfo(inputPathEdit_->text()).path() + "/output/" + QFileInfo(inputPathEdit_->text()).completeBaseName() + "_archive_restore_1440p.mp4");
    appendLog("Applied template: " + quickTemplateCombo_->currentText());
    quickTemplateCombo_->setCurrentIndex(0);
}

// ─────────────────────────────────────────────────────────────────
// Drag-and-drop
// ─────────────────────────────────────────────────────────────────

void MainWindow::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) event->acceptProposedAction();
    else event->ignore();
}

void MainWindow::dropEvent(QDropEvent* event) {
    const auto urls = event->mimeData()->urls();
    if (urls.isEmpty()) { event->ignore(); return; }
    const QString path = urls.first().toLocalFile();
    if (!path.isEmpty()) {
        inputPathEdit_->setText(path);
        if (outputPathEdit_->text().trimmed().isEmpty()) {
            const QFileInfo fi(path);
            outputPathEdit_->setText(fi.path() + "/output/" + fi.completeBaseName() + "_enhanced.mp4");
        }
        appendLog("Input loaded by drag-and-drop: " + path);
    }
    refreshCommandPreview();
    event->acceptProposedAction();
}
