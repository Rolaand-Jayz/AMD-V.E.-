#include "main_window.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <QApplication>
#include <QClipboard>
#include <QCloseEvent>
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
#include <QImage>
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
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QTabWidget>
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
#include "filter_browser.hpp"
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

constexpr int kRoleModelId = Qt::UserRole;
constexpr int kRoleAddEnabled = Qt::UserRole + 1;
constexpr int kRoleNeedsCompile = Qt::UserRole + 2;
constexpr int kRoleStatusMessage = Qt::UserRole + 3;
constexpr int kRoleStageKind = Qt::UserRole + 4;

struct ModelFamilyInfo {
    const char* id;
    const char* name;
};

ModelFamilyInfo modelFamilyInfoForId(const std::string& modelId) {
    static const std::unordered_map<std::string, ModelFamilyInfo> map = {
        {"nomos2-otf-x4", {"nomos2-otf", "Nomos2 OTF ESRGAN"}},
        {"realesrgan-x4-restore", {"realesrgan-x4", "Real-ESRGAN x4"}},
        {"realesrgan-x4-general", {"realesrgan-x4", "Real-ESRGAN x4"}},
        {"realesrgan-x4plus-pth", {"realesrgan-x4plus-pth", "Real-ESRGAN x4+ (PyTorch)"}},
        {"realesrgan-x4-axera", {"realesrgan-x4-axera", "Real-ESRGAN x4 (AXERA export)"}},
        {"realesrnet-x2plus", {"realesrnet-x2plus", "Real-ESRNet x2+"}},
        {"nmkd-siax-x4", {"nmkd-siax-x4", "NMKD Siax 200k"}},
        {"ultrasharp-x4", {"ultrasharp-x4", "UltraSharp x4"}},
        {"wtp-uds-esrgan-x4", {"wtp-uds-esrgan-x4", "WTP-UDS-ESRGAN x4"}},
        {"nafnet-denoise", {"nafnet-restoration", "NAFNet Blind Restoration"}},
        {"nafnet-deblur-gopro", {"nafnet-restoration", "NAFNet Blind Restoration"}},
        {"nafnet-dehalo", {"nafnet-restoration", "NAFNet Blind Restoration"}},
        {"clearreality-x4-denoise", {"clearreality-v1", "ClearReality V1"}},
        {"clearreality-x4-fast", {"clearreality-v1", "ClearReality V1"}},
        {"remacri-x4", {"remacri-x4", "Remacri x4"}},
        {"ultrasharpv2-deblur", {"ultrasharpv2", "UltraSharp V2"}},
        {"ultrasharpv2-x4", {"ultrasharpv2", "UltraSharp V2"}},
        {"ultrasharpv2-lite-x4", {"ultrasharpv2-lite", "UltraSharp V2 Lite"}},
        {"iqa-color-enhance", {"parametric-color-fix", "Parametric Color Fix"}},
        {"swinir-color", {"swinir-x4", "SwinIR x4"}},
        {"swinir-x4-general", {"swinir-x4", "SwinIR x4"}},
        {"animesharp-x4", {"animesharp-x4", "AnimeSharp x4"}},
        {"modernspanimation-x2", {"modernspanimation-v2", "ModernSpanimation v2 x2"}},
        {"modernspanimation-x2-fp32", {"modernspanimation-v2-fp32", "ModernSpanimation v2 x2 fp32"}},
        {"modernspanimation-x2-v1compact", {"modernspanimation-v1-compact", "ModernSpanimation v1 Compact x2"}},
        {"animejananai-hd-compact-x2", {"animejanai-hd-v3", "AnimeJaNai HD V3 Compact x2"}},
        {"realistic-rescaler-x4", {"realistic-rescaler-x4", "RealisticRescaler x4"}},
        {"sharpen-cas", {"cas-sharpen", "Contrast-Adaptive Sharpening"}},
        {"rife-v4-7", {"rife-v4-7", "RIFE v4.7"}},
        {"rife-v4-8", {"rife-v4-8", "RIFE v4.8"}},
        {"rife-v4-9", {"rife-v4-9", "RIFE v4.9"}},
        {"interp-ffmpeg", {"interp-ffmpeg", "FFmpeg Interpolation"}}
    };
    const auto it = map.find(modelId);
    if (it != map.end()) {
        return it->second;
    }
    return {nullptr, nullptr};
}

QString modelFamilyId(const std::string& modelId) {
    const auto info = modelFamilyInfoForId(modelId);
    return info.id ? QString::fromUtf8(info.id) : QString::fromStdString(modelId);
}

QString modelFamilyName(const std::string& modelId, const std::string& fallback) {
    const auto info = modelFamilyInfoForId(modelId);
    return info.name ? QString::fromUtf8(info.name) : QString::fromStdString(fallback);
}

QString joinStageTitles(const QStringList& stages) {
    if (stages.isEmpty()) {
        return QStringLiteral("No enhancements");
    }
    return stages.join(QStringLiteral(", "));
}

bool hasMxrExtension(const std::string& path) {
    if (path.size() < 4) { return false; }
    const std::string ext = path.substr(path.size() - 4);
    return ext == ".mxr" || ext == ".MXR";
}

bool hasUsableModelPath(const std::string& path) {
    return !path.empty() && path != "(builtin)";
}

bool isMigraphxCompatibleFormat(ave::ModelFormat format) {
    return format == ave::ModelFormat::Onnx || format == ave::ModelFormat::Pytorch;
}

bool isNcnnCompatibleFormat(ave::ModelFormat format) {
    return format == ave::ModelFormat::NcnnBin;
}

bool isCompatibleWithBackend(const ave::ManagedModel& model, ave::BackendType backend) {
    if (backend == ave::BackendType::MiGraphX) {
        return (isMigraphxCompatibleFormat(model.entry.sourceFormat) &&
                hasUsableModelPath(model.downloadedPath) &&
                !hasMxrExtension(model.downloadedPath)) ||
               (hasUsableModelPath(model.downloadedPath) && hasMxrExtension(model.downloadedPath)) ||
               (hasUsableModelPath(model.convertedPath) && hasMxrExtension(model.convertedPath));
    }
    if (backend == ave::BackendType::NcnnVulkan) {
        return isNcnnCompatibleFormat(model.entry.sourceFormat);
    }
    if (backend == ave::BackendType::Auto) {
        return isCompatibleWithBackend(model, ave::BackendType::MiGraphX) ||
               isCompatibleWithBackend(model, ave::BackendType::NcnnVulkan);
    }
    return false;
}

bool isReadyForBackend(const ave::ManagedModel& model, ave::BackendType backend) {
    if (backend == ave::BackendType::MiGraphX) {
        return (hasUsableModelPath(model.downloadedPath) && hasMxrExtension(model.downloadedPath)) ||
               (hasUsableModelPath(model.convertedPath) && hasMxrExtension(model.convertedPath));
    }
    if (backend == ave::BackendType::NcnnVulkan) {
        return model.state == ave::ModelState::Downloaded &&
               isNcnnCompatibleFormat(model.entry.sourceFormat) &&
               !model.downloadedPath.empty();
    }
    if (backend == ave::BackendType::Auto) {
        return isReadyForBackend(model, ave::BackendType::MiGraphX) ||
               isReadyForBackend(model, ave::BackendType::NcnnVulkan);
    }
    return false;
}

bool needsMigraphxCompile(const ave::ManagedModel& model, ave::BackendType backend) {
    if (backend != ave::BackendType::MiGraphX && backend != ave::BackendType::Auto) {
        return false;
    }
    return isMigraphxCompatibleFormat(model.entry.sourceFormat) &&
           hasUsableModelPath(model.downloadedPath) &&
           model.state == ave::ModelState::Downloaded &&
           !hasMxrExtension(model.downloadedPath) &&
           !hasMxrExtension(model.convertedPath);
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
    resize(1440, 920);
    setAcceptDrops(true);

    appSettings_.load();
    buildUi();
    applySettingsToUi(true);
    wireActions();

    modelManager_.refresh();
    refreshModelFamilies();
    refreshParamPanel();
    refreshStageBuilderActions();
    refreshRequestedStages();
    refreshPlannedStages();
    refreshActiveFilters();
    refreshCommandPreview();
}

void MainWindow::closeEvent(QCloseEvent* event) {
    persistUiStateToSettings();
    QMainWindow::closeEvent(event);
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
        storeSelectedFamilyCapabilityDraft();
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
            l->setText(sliderPct(v));
            storeSelectedFamilyCapabilityDraft();
            refreshCommandPreview();
        });
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
    connect(upscaleWidthSpin_,  &QSpinBox::valueChanged, this, [this](int) {
        storeSelectedFamilyCapabilityDraft();
        refreshCommandPreview();
    });
    connect(upscaleHeightSpin_, &QSpinBox::valueChanged, this, [this](int) {
        storeSelectedFamilyCapabilityDraft();
        refreshCommandPreview();
    });
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
        sharpenLabel_->setText(sliderPct(v));
        storeSelectedFamilyCapabilityDraft();
        refreshCommandPreview();
    });
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
    connect(interpolateFpsSpin_, &QSpinBox::valueChanged, this, [this](int) {
        storeSelectedFamilyCapabilityDraft();
        refreshCommandPreview();
    });
    connect(sceneDetectToggle_, &ToggleSwitch::toggled,   this, [this](bool) {
        storeSelectedFamilyCapabilityDraft();
        refreshCommandPreview();
    });
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
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(10);

    auto* header = new QGroupBox(central);
    header->setTitle(QString());
    auto* headerLayout = new QHBoxLayout(header);
    headerLayout->setContentsMargins(14, 10, 14, 10);
    headerLayout->setSpacing(10);

    auto* headingLayout = new QVBoxLayout;
    headingLayout->setSpacing(2);
    auto* titleLabel = new QLabel("AMD Video Enhancer", header);
    titleLabel->setStyleSheet("font-size: 20px; font-weight: 700;");
    auto* subtitleLabel = new QLabel(
        "1. Set up the job.  2. Build the pipeline.  3. Review the command.  4. Preview or run.",
        header);
    subtitleLabel->setWordWrap(true);
    subtitleLabel->setStyleSheet("color: #666;");
    headingLayout->addWidget(titleLabel);
    headingLayout->addWidget(subtitleLabel);
    headerLayout->addLayout(headingLayout, 1);

    auto* modelMgrBtn = new QPushButton("Model Manager", header);
    auto* probeBtn = new QPushButton("Probe Backends", header);
    auto* settingsBtn = new QPushButton("Settings", header);
    auto* headerButtons = new QHBoxLayout;
    headerButtons->setContentsMargins(0, 0, 0, 0);
    headerButtons->setSpacing(6);
    headerButtons->addWidget(modelMgrBtn);
    headerButtons->addWidget(probeBtn);
    headerButtons->addWidget(settingsBtn);
    headerLayout->addLayout(headerButtons);
    root->addWidget(header);

    auto* workspaceSplit = new QSplitter(Qt::Horizontal, central);
    workspaceSplit->setChildrenCollapsible(false);

    auto* leftPane = new QWidget(workspaceSplit);
    auto* leftLayout = new QVBoxLayout(leftPane);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(10);

    auto* rightPane = new QWidget(workspaceSplit);
    auto* rightLayout = new QVBoxLayout(rightPane);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(10);

    // ── Step 1: job setup ───────────────────────────────────────
    auto* setupGroup = new QGroupBox("1. Job Setup", leftPane);
    auto* setupGrid = new QGridLayout(setupGroup);
    setupGrid->setHorizontalSpacing(10);
    setupGrid->setVerticalSpacing(8);
    setupGrid->setColumnStretch(1, 1);
    setupGrid->setColumnStretch(3, 1);

    auto* setupHint = new QLabel(
        "Choose the source video, output path, backend, and encode defaults for this run.",
        setupGroup);
    setupHint->setWordWrap(true);
    setupHint->setStyleSheet("color: #666;");
    setupGrid->addWidget(setupHint, 0, 0, 1, 4);

    setupGrid->addWidget(new QLabel("Input", setupGroup), 1, 0);
    inputPathEdit_ = new QLineEdit(setupGroup);
    inputPathEdit_->setPlaceholderText("Drop a video here or browse to one on disk");
    setupGrid->addWidget(inputPathEdit_, 1, 1, 1, 2);
    auto* browseIn = new QPushButton("Browse...", setupGroup);
    setupGrid->addWidget(browseIn, 1, 3);

    setupGrid->addWidget(new QLabel("Output", setupGroup), 2, 0);
    outputPathEdit_ = new QLineEdit(setupGroup);
    outputPathEdit_->setPlaceholderText("Select where the enhanced video should be written");
    setupGrid->addWidget(outputPathEdit_, 2, 1, 1, 2);
    auto* browseOut = new QPushButton("Browse...", setupGroup);
    setupGrid->addWidget(browseOut, 2, 3);

    backendCombo_ = new QComboBox(setupGroup);
    backendCombo_->addItem("Auto (MiGraphX → NCNN → Vulkan)", static_cast<int>(ave::BackendType::Auto));
    backendCombo_->addItem("MiGraphX (ROCm)",         static_cast<int>(ave::BackendType::MiGraphX));
    backendCombo_->addItem("NCNN (Vulkan)",            static_cast<int>(ave::BackendType::NcnnVulkan));
    backendCombo_->addItem("Vulkan Compute",           static_cast<int>(ave::BackendType::VulkanCompute));
    backendCombo_->addItem("VapourSynth",              static_cast<int>(ave::BackendType::VapourSynth));
    backendCombo_->addItem("GLSL Shader",              static_cast<int>(ave::BackendType::GlslShader));
    setupGrid->addWidget(new QLabel("Backend", setupGroup), 3, 0);
    setupGrid->addWidget(backendCombo_, 3, 1);

    codecCombo_ = new QComboBox(setupGroup);
    codecCombo_->addItems({
        "libx264", "libx265", "libsvtav1",
        "h264_vaapi", "hevc_vaapi", "av1_vaapi",
        "h264_amf", "hevc_amf", "av1_amf",
        "libvpx-vp9", "libaom-av1", "ffv1", "utvideo"
    });
    codecCombo_->setEditable(true);  // allow custom codecs
    codecCombo_->setCurrentText("libx264");
    setupGrid->addWidget(new QLabel("Codec", setupGroup), 3, 2);
    setupGrid->addWidget(codecCombo_, 3, 3);

    profileCombo_ = new QComboBox(setupGroup);
    profileCombo_->setEditable(true);
    setupGrid->addWidget(new QLabel("Profile", setupGroup), 4, 0);
    setupGrid->addWidget(profileCombo_, 4, 1);

    presetCombo_ = new QComboBox(setupGroup);
    presetCombo_->setEditable(true);
    setupGrid->addWidget(new QLabel("Preset", setupGroup), 4, 2);
    setupGrid->addWidget(presetCombo_, 4, 3);

    // Populate profile and preset choices based on codec selection.
    auto populateCodecOptions = [this]() {
        const QString codec = codecCombo_->currentText().trimmed();
        const QString prevProfile = profileCombo_->currentText();
        const QString prevPreset  = presetCombo_->currentText();

        profileCombo_->clear();
        presetCombo_->clear();

        if (codec.contains("x264") || codec.contains("h264")) {
            profileCombo_->addItems({"(auto)", "baseline", "main", "high", "high10", "high422", "high444"});
            presetCombo_->addItems({"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo"});
        } else if (codec.contains("x265") || codec.contains("hevc")) {
            profileCombo_->addItems({"(auto)", "main", "main10", "main12", "main422-10", "main444-10"});
            presetCombo_->addItems({"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow","placebo"});
        } else if (codec.contains("svtav1") || codec.contains("aom") || codec.contains("av1")) {
            profileCombo_->addItems({"(auto)", "main", "high", "professional"});
            // SVT-AV1 / libaom use numeric presets 0-13
            for (int i = 0; i <= 13; ++i) presetCombo_->addItem(QString::number(i));
        } else if (codec.contains("vpx") || codec.contains("vp9")) {
            profileCombo_->addItems({"(auto)", "0", "1", "2", "3"});
            presetCombo_->addItems({"realtime","good","best"});
        } else {
            profileCombo_->addItem("(auto)");
            presetCombo_->addItems({"ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"});
        }

        // Restore previous selection if still valid.
        int pi = profileCombo_->findText(prevProfile);
        profileCombo_->setCurrentIndex(pi >= 0 ? pi : 0);
        int si = presetCombo_->findText(prevPreset);
        presetCombo_->setCurrentIndex(si >= 0 ? si : presetCombo_->findText("medium"));
        if (presetCombo_->currentIndex() < 0) presetCombo_->setCurrentIndex(0);
    };
    populateCodecOptions();
    connect(codecCombo_, &QComboBox::currentTextChanged, this, [populateCodecOptions](const QString&) {
        populateCodecOptions();
    });

    crfSpin_ = new QSpinBox(setupGroup);
    crfSpin_->setRange(0, 63);
    crfSpin_->setValue(18);
    setupGrid->addWidget(new QLabel("CRF", setupGroup), 5, 0);
    setupGrid->addWidget(crfSpin_, 5, 1);

    dryRunToggle_ = new ToggleSwitch(setupGroup);
    {
        auto* dryRow = new QHBoxLayout;
        dryRow->setContentsMargins(0, 0, 0, 0);
        dryRow->addWidget(new QLabel("Dry run:", setupGroup));
        dryRow->addWidget(dryRunToggle_);
        dryRow->addStretch();
        setupGrid->addLayout(dryRow, 5, 2, 1, 2);
    }

    auto* saveBtn = new QPushButton("Save Profile", setupGroup);
    auto* loadBtn = new QPushButton("Load Profile", setupGroup);
    auto* profileButtons = new QHBoxLayout;
    profileButtons->setContentsMargins(0, 0, 0, 0);
    profileButtons->addStretch();
    profileButtons->addWidget(saveBtn);
    profileButtons->addWidget(loadBtn);
    setupGrid->addLayout(profileButtons, 6, 0, 1, 4);
    leftLayout->addWidget(setupGroup);

    connect(modelMgrBtn,  &QPushButton::clicked, this, &MainWindow::openModelManager);
    connect(probeBtn,     &QPushButton::clicked, this, &MainWindow::probeBackends);
    connect(settingsBtn,  &QPushButton::clicked, this, &MainWindow::openSettings);
    connect(saveBtn,      &QPushButton::clicked, this, &MainWindow::saveProfile);
    connect(loadBtn,      &QPushButton::clicked, this, &MainWindow::loadProfile);

    connect(browseIn, &QPushButton::clicked, this, [this]() {
        const QString p = QFileDialog::getOpenFileName(this, "Select Input Video", inputPathEdit_->text());
        if (!p.isEmpty()) inputPathEdit_->setText(p);
    });
    connect(browseOut, &QPushButton::clicked, this, [this]() {
        const QString p = QFileDialog::getSaveFileName(this, "Select Output Video", outputPathEdit_->text());
        if (!p.isEmpty()) outputPathEdit_->setText(p);
    });

    // ── Step 2: pipeline builder ────────────────────────────────
    auto* pipelineGroup = new QGroupBox("2. Build Enhancement Pipeline", leftPane);
    auto* pipelineLayout = new QVBoxLayout(pipelineGroup);
    pipelineLayout->setSpacing(8);

    auto* templateRow = new QHBoxLayout;
    auto* pipelineHint = new QLabel(
        "Start with a template or assemble the stage list manually. Filters are available in a separate tab.",
        pipelineGroup);
    pipelineHint->setWordWrap(true);
    pipelineHint->setStyleSheet("color: #666;");
    templateRow->addWidget(pipelineHint, 1);
    quickTemplateCombo_ = new QComboBox(pipelineGroup);
    quickTemplateCombo_->addItem("Start from template");
    quickTemplateCombo_->addItem("Web Cleanup 1080p60");
    quickTemplateCombo_->addItem("Anime Upscale 4K60");
    quickTemplateCombo_->addItem("Archive Restore 1440p");
    templateRow->addWidget(quickTemplateCombo_);
    pipelineLayout->addLayout(templateRow);

    auto* pipelineTabs = new QTabWidget(pipelineGroup);

    auto* stagesTab = new QWidget(pipelineTabs);
    auto* stagesTabLayout = new QVBoxLayout(stagesTab);
    stagesTabLayout->setContentsMargins(0, 0, 0, 0);
    stagesTabLayout->setSpacing(8);

    auto* sbGroup = new QGroupBox("Add Model Family", stagesTab);
    auto* sbGrid  = new QGridLayout(sbGroup);
    sbGrid->setHorizontalSpacing(10);
    sbGrid->setVerticalSpacing(8);
    sbGrid->setColumnStretch(1, 1);
    sbGrid->setColumnStretch(3, 1);

    modelFamilyCombo_ = new QComboBox(sbGroup);
    modelFamilyCombo_->setMinimumWidth(320);
    sbGrid->addWidget(new QLabel("Model family:"), 0, 0);
    sbGrid->addWidget(modelFamilyCombo_, 0, 1, 1, 3);

    auto* capabilityGroup = new QGroupBox("Enhancements Offered", sbGroup);
    auto* capabilityLayout = new QVBoxLayout(capabilityGroup);
    auto* capabilityHint = new QLabel(
        "Pick a family once, then toggle the enhancement roles this family can cover.",
        capabilityGroup);
    capabilityHint->setWordWrap(true);
    capabilityHint->setStyleSheet("color: #666;");
    capabilityLayout->addWidget(capabilityHint);
    familyCapabilitiesView_ = new QListWidget(capabilityGroup);
    familyCapabilitiesView_->setAlternatingRowColors(true);
    familyCapabilitiesView_->setSelectionMode(QAbstractItemView::SingleSelection);
    capabilityLayout->addWidget(familyCapabilitiesView_, 1);
    sbGrid->addWidget(capabilityGroup, 1, 0, 1, 2);

    auto* editorGroup = new QGroupBox("Adjust Selected Enhancement", sbGroup);
    auto* editorLayout = new QVBoxLayout(editorGroup);
    capabilityEditorLabel_ = new QLabel(
        "Select an enhancement on the left to adjust its parameters.",
        editorGroup);
    capabilityEditorLabel_->setWordWrap(true);
    capabilityEditorLabel_->setStyleSheet("color: #666;");
    editorLayout->addWidget(capabilityEditorLabel_);

    paramStack_ = new QStackedWidget(editorGroup);
    paramStack_->addWidget(buildStrengthPanel(editorGroup));    // 0: generic strength
    paramStack_->addWidget(buildColorFixPanel(editorGroup));    // 1: colorfix
    paramStack_->addWidget(buildUpscalePanel(editorGroup));     // 2: upscale
    paramStack_->addWidget(buildSharpenPanel(editorGroup));     // 3: sharpen
    paramStack_->addWidget(buildInterpolatePanel(editorGroup)); // 4: interpolate
    paramStack_->addWidget(buildEmptyPanel(editorGroup));       // 5: empty fallback
    editorLayout->addWidget(paramStack_);

    extraParamsEdit_ = new QLineEdit(editorGroup);
    extraParamsEdit_->setPlaceholderText("extra_key=value, another=42");
    editorLayout->addWidget(new QLabel("Extra params for selected enhancement:", editorGroup));
    editorLayout->addWidget(extraParamsEdit_);
    sbGrid->addWidget(editorGroup, 1, 2, 1, 2);

    modelStatusLabel_ = new QLabel(sbGroup);
    modelStatusLabel_->setStyleSheet("color: #666;");
    modelStatusLabel_->setWordWrap(true);
    sbGrid->addWidget(modelStatusLabel_, 2, 0, 1, 4);

    compileModelButton_ = new QPushButton("Compile/Manage Family Models...", sbGroup);
    compileModelButton_->setVisible(false);
    sbGrid->addWidget(compileModelButton_, 3, 0, 1, 4);

    addStageButton_ = new QPushButton("Add Selected Enhancements", sbGroup);
    removeStageButton_ = new QPushButton("Remove", sbGroup);
    moveUpButton_ = new QPushButton("Move Up", sbGroup);
    moveDownButton_ = new QPushButton("Move Down", sbGroup);
    clearStagesButton_ = new QPushButton("Clear All", sbGroup);

    auto* stageRow = new QHBoxLayout;
    stageRow->addWidget(addStageButton_);
    stageRow->addWidget(removeStageButton_);
    stageRow->addWidget(moveUpButton_);
    stageRow->addWidget(moveDownButton_);
    stageRow->addWidget(clearStagesButton_);
    stageRow->addStretch();
    sbGrid->addLayout(stageRow, 4, 0, 1, 4);
    stagesTabLayout->addWidget(sbGroup);

    connect(addStageButton_,    &QPushButton::clicked, this, &MainWindow::addStage);
    connect(compileModelButton_,&QPushButton::clicked, this, &MainWindow::compileSelectedModel);
    connect(removeStageButton_, &QPushButton::clicked, this, &MainWindow::removeSelectedStage);
    connect(moveUpButton_,      &QPushButton::clicked, this, [this]() { moveSelectedStage(-1); });
    connect(moveDownButton_,    &QPushButton::clicked, this, [this]() { moveSelectedStage(1); });
    connect(clearStagesButton_, &QPushButton::clicked, this, &MainWindow::clearStages);

    auto* plannerHint = new QLabel(
        "Requested order is editable. The planner preview shows the execution order the app will actually use.",
        stagesTab);
    plannerHint->setWordWrap(true);
    plannerHint->setStyleSheet("color: #666;");
    stagesTabLayout->addWidget(plannerHint);

    auto* listSplit = new QSplitter(Qt::Horizontal, stagesTab);
    listSplit->setChildrenCollapsible(false);

    auto* reqGroup  = new QGroupBox("Requested Order", listSplit);
    auto* reqLayout = new QVBoxLayout(reqGroup);
    requestedStagesView_ = new QListWidget(reqGroup);
    requestedStagesView_->setDragDropMode(QAbstractItemView::InternalMove);
    requestedStagesView_->setDefaultDropAction(Qt::MoveAction);
    requestedStagesView_->setSelectionMode(QAbstractItemView::SingleSelection);
    requestedStagesView_->setAlternatingRowColors(true);
    reqLayout->addWidget(requestedStagesView_);

    auto* planGroup  = new QGroupBox("Planner Order", listSplit);
    auto* planLayout = new QVBoxLayout(planGroup);
    plannedStagesView_ = new QListWidget(planGroup);
    plannedStagesView_->setAlternatingRowColors(true);
    plannedStagesView_->setSelectionMode(QAbstractItemView::NoSelection);
    planLayout->addWidget(plannedStagesView_);

    listSplit->addWidget(reqGroup);
    listSplit->addWidget(planGroup);
    listSplit->setStretchFactor(0, 1);
    listSplit->setStretchFactor(1, 1);
    stagesTabLayout->addWidget(listSplit, 1);

    pipelineTabs->addTab(stagesTab, "Stages");

    auto* filtersTab = new QWidget(pipelineTabs);
    auto* filtersTabLayout = new QVBoxLayout(filtersTab);
    filtersTabLayout->setContentsMargins(0, 0, 0, 0);
    filtersTabLayout->setSpacing(8);
    auto* filtersHint = new QLabel(
        "Enable catalog filters on the left. The applied list on the right shows what will be carried into the job.",
        filtersTab);
    filtersHint->setWordWrap(true);
    filtersHint->setStyleSheet("color: #666;");
    filtersTabLayout->addWidget(filtersHint);

    auto* filtersSplit = new QSplitter(Qt::Horizontal, filtersTab);
    filtersSplit->setChildrenCollapsible(false);

    auto* filterCatalogPane = new QWidget(filtersSplit);
    auto* filterCatalogLayout = new QVBoxLayout(filterCatalogPane);
    filterCatalogLayout->setContentsMargins(0, 0, 0, 0);
    filterCatalogLayout->setSpacing(6);
    filterBrowser_ = new FilterBrowser(filterCatalogPane);
    filterCatalogLayout->addWidget(filterBrowser_, 1);

    auto* activeFiltersGroup = new QGroupBox("Applied Filters", filtersSplit);
    auto* activeFiltersLayout = new QVBoxLayout(activeFiltersGroup);
    auto* activeFiltersHint = new QLabel(
        "Selected filters are listed here so they are visible like the main enhancement stages.",
        activeFiltersGroup);
    activeFiltersHint->setWordWrap(true);
    activeFiltersHint->setStyleSheet("color: #666;");
    activeFiltersLayout->addWidget(activeFiltersHint);
    activeFiltersView_ = new QListWidget(activeFiltersGroup);
    activeFiltersView_->setAlternatingRowColors(true);
    activeFiltersView_->setSelectionMode(QAbstractItemView::NoSelection);
    activeFiltersLayout->addWidget(activeFiltersView_, 1);

    filtersSplit->addWidget(filterCatalogPane);
    filtersSplit->addWidget(activeFiltersGroup);
    filtersSplit->setStretchFactor(0, 2);
    filtersSplit->setStretchFactor(1, 1);
    filtersTabLayout->addWidget(filtersSplit, 1);
    pipelineTabs->addTab(filtersTab, "Filters");

    pipelineLayout->addWidget(pipelineTabs, 1);
    leftLayout->addWidget(pipelineGroup, 1);

    // ── Step 3: review + run ────────────────────────────────────
    auto* reviewGroup = new QGroupBox("3. Review + Run", rightPane);
    auto* reviewLayout = new QVBoxLayout(reviewGroup);
    reviewLayout->setSpacing(8);

    auto* reviewHint = new QLabel(
        "Check the generated CLI, run a short preview clip if needed, then launch the full job.",
        reviewGroup);
    reviewHint->setWordWrap(true);
    reviewHint->setStyleSheet("color: #666;");
    reviewLayout->addWidget(reviewHint);

    auto* commandRow = new QHBoxLayout;
    commandPreviewEdit_ = new QLineEdit(reviewGroup);
    commandPreviewEdit_->setReadOnly(true);
    auto* copyBtn = new QPushButton("Copy Command", reviewGroup);
    commandRow->addWidget(commandPreviewEdit_, 1);
    commandRow->addWidget(copyBtn);
    reviewLayout->addLayout(commandRow);

    auto* actionRow = new QHBoxLayout;
    runButton_ = new QPushButton("Run Job", reviewGroup);
    runButton_->setStyleSheet(
        "QPushButton { font-weight: 600; background-color: #1f6feb; color: white; padding: 5px 14px; }");

    previewButton_ = new QPushButton("Preview Clip", reviewGroup);
    previewButton_->setStyleSheet(
        "QPushButton { font-weight: 600; background-color: #0f766e; color: white; padding: 5px 14px; }");
    previewButton_->setToolTip("Process a short preview clip to see the result before the full run");

    pauseButton_ = new QPushButton("Pause", reviewGroup);
    pauseButton_->setToolTip("Pause or resume processing");
    pauseButton_->setVisible(false);

    cancelButton_ = new QPushButton("Cancel", reviewGroup);
    cancelButton_->setStyleSheet(
        "QPushButton { background-color: #b42318; color: white; padding: 5px 14px; }");
    cancelButton_->setToolTip("Cancel the current processing job");
    cancelButton_->setVisible(false);

    previewDurationSpin_ = new QSpinBox(reviewGroup);
    previewDurationSpin_->setRange(1, 60);
    previewDurationSpin_->setValue(10);
    previewDurationSpin_->setSuffix(" sec");
    previewDurationSpin_->setToolTip("Preview clip duration in seconds");

    actionRow->addWidget(runButton_);
    actionRow->addWidget(previewButton_);
    actionRow->addWidget(pauseButton_);
    actionRow->addWidget(cancelButton_);
    actionRow->addStretch();
    actionRow->addWidget(new QLabel("Preview length:", reviewGroup));
    actionRow->addWidget(previewDurationSpin_);
    reviewLayout->addLayout(actionRow);

    progressBar_ = new QProgressBar(reviewGroup);
    progressBar_->setRange(0, 100);
    progressBar_->setFormat("Overall: %p%");
    progressBar_->setVisible(false);

    taskProgressBar_ = new QProgressBar(reviewGroup);
    taskProgressBar_->setRange(0, 100);
    taskProgressBar_->setFormat("Task: %p%");
    taskProgressBar_->setVisible(false);

    progressLabel_ = new QLabel(reviewGroup);
    progressLabel_->setWordWrap(true);
    progressLabel_->setStyleSheet("color: #555; font-style: italic;");
    progressLabel_->setVisible(false);

    reviewLayout->addWidget(new QLabel("Overall progress", reviewGroup));
    reviewLayout->addWidget(progressBar_);
    reviewLayout->addWidget(new QLabel("Current task", reviewGroup));
    reviewLayout->addWidget(taskProgressBar_);
    reviewLayout->addWidget(progressLabel_);
    rightLayout->addWidget(reviewGroup);

    connect(runButton_, &QPushButton::clicked, this, &MainWindow::runJob);
    connect(previewButton_, &QPushButton::clicked, this, &MainWindow::runPreview);
    connect(pauseButton_, &QPushButton::clicked, this, [this]() {
        const bool wasPaused = pauseFlag_.load();
        pauseFlag_.store(!wasPaused);
        if (wasPaused) {
            pauseButton_->setText("Pause");
            appendLog("Processing resumed.");
            progressLabel_->setText("Resumed…");
        } else {
            pauseButton_->setText("Resume");
            appendLog("Processing paused.");
            progressLabel_->setText("Paused — click Resume to continue");
        }
    });
    connect(cancelButton_, &QPushButton::clicked, this, [this]() {
        cancelFlag_.store(true);
        pauseFlag_.store(false);   // unpause so the loop can exit
        appendLog("Cancel requested — stopping after current frame…");
        cancelButton_->setEnabled(false);
        pauseButton_->setEnabled(false);
        progressLabel_->setText("Cancelling…");
    });
    connect(copyBtn, &QPushButton::clicked, this, [this]() {
        if (auto* cb = QApplication::clipboard())
            cb->setText(commandPreviewEdit_->text());
        appendLog("Command copied to clipboard.");
    });
    connect(quickTemplateCombo_, &QComboBox::currentIndexChanged, this, &MainWindow::applyQuickTemplate);

    auto* monitorSplit = new QSplitter(Qt::Vertical, rightPane);
    monitorSplit->setChildrenCollapsible(false);

    // ── Step 4: frame preview ───────────────────────────────────
    auto* previewGroup  = new QGroupBox("4. Frame Preview", monitorSplit);
    auto* previewLayout = new QVBoxLayout(previewGroup);
    framePreviewLabel_ = new QLabel(previewGroup);
    framePreviewLabel_->setAlignment(Qt::AlignCenter);
    framePreviewLabel_->setMinimumSize(320, 180);
    framePreviewLabel_->setMaximumHeight(360);
    framePreviewLabel_->setStyleSheet("QLabel { background-color: #1a1a2e; color: #888; border: 1px solid #333; }");
    framePreviewLabel_->setText("Frame preview will appear here during processing");
    framePreviewLabel_->setScaledContents(false);
    previewLayout->addWidget(framePreviewLabel_);
    monitorSplit->addWidget(previewGroup);

    // ── Step 5: session log ─────────────────────────────────────
    auto* logGroup  = new QGroupBox("5. Session Log", monitorSplit);
    auto* logLayout = new QVBoxLayout(logGroup);
    logView_ = new QPlainTextEdit(logGroup);
    logView_->setReadOnly(true);
    logView_->setMaximumBlockCount(2000);
    logLayout->addWidget(logView_);
    monitorSplit->addWidget(logGroup);
    monitorSplit->setStretchFactor(0, 1);
    monitorSplit->setStretchFactor(1, 1);
    rightLayout->addWidget(monitorSplit, 1);

    workspaceSplit->addWidget(leftPane);
    workspaceSplit->addWidget(rightPane);
    workspaceSplit->setStretchFactor(0, 3);
    workspaceSplit->setStretchFactor(1, 2);
    root->addWidget(workspaceSplit, 1);
}

// ─────────────────────────────────────────────────────────────────
// wireActions
// ─────────────────────────────────────────────────────────────────

void MainWindow::wireActions() {
    auto ref = [this]() { refreshCommandPreview(); };

    connect(inputPathEdit_,  &QLineEdit::textChanged, this, [this, ref](const QString& text) {
        if (text.trimmed().isEmpty()) {
            if (outputPathAutoManaged_) {
                setManagedOutputPath(QString(), true);
            }
        } else {
            applySuggestedOutputPath(text, false);
        }
        ref();
    });
    connect(outputPathEdit_, &QLineEdit::textChanged, this, [this, ref](const QString& text) {
        if (!updatingOutputPath_) {
            outputPathAutoManaged_ = text.trimmed().isEmpty();
        }
        ref();
    });
    connect(codecCombo_,     &QComboBox::currentTextChanged, this, [ref](const QString&) { ref(); });
    connect(profileCombo_,   &QComboBox::currentTextChanged, this, [ref](const QString&) { ref(); });
    connect(presetCombo_,    &QComboBox::currentTextChanged, this, [ref](const QString&) { ref(); });
    connect(crfSpin_,        &QSpinBox::valueChanged, this, [ref](int)           { ref(); });
    connect(backendCombo_,   &QComboBox::currentIndexChanged, this, [this, ref](int) {
        refreshModelFamilies();
        refreshParamPanel();
        refreshStageBuilderActions();
        ref();
    });
    connect(dryRunToggle_,   &ToggleSwitch::toggled,  this, [ref](bool)          { ref(); });
    connect(extraParamsEdit_,&QLineEdit::textChanged, this, [this, ref](const QString&) {
        storeSelectedFamilyCapabilityDraft();
        ref();
    });
    connect(modelFamilyCombo_, &QComboBox::currentIndexChanged, this, [this, ref](int) {
        refreshModelFamilies();
        refreshParamPanel();
        refreshStageBuilderActions();
        ref();
    });
    connect(familyCapabilitiesView_, &QListWidget::currentRowChanged, this, [this, ref](int) {
        storeSelectedFamilyCapabilityDraft();
        refreshParamPanel();
        refreshStageBuilderActions();
        ref();
    });
    connect(familyCapabilitiesView_, &QListWidget::itemChanged, this, [this, ref](QListWidgetItem*) {
        refreshStageBuilderActions();
        ref();
    });
    connect(filterBrowser_,  &FilterBrowser::filtersChanged, this, [this]() {
        refreshActiveFilters();
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

void MainWindow::refreshModelFamilies() {
    if (!modelFamilyCombo_ || !familyCapabilitiesView_) {
        return;
    }

    storeSelectedFamilyCapabilityDraft();

    const QString requestedFamilyId = modelFamilyCombo_->currentData().toString();
    const auto requestedCapability = activeFamilyCapability_;
    const auto previousDrafts = familyDraftStages_;
    const auto backend = static_cast<ave::BackendType>(backendCombo_->currentData().toInt());

    struct FamilyBucket {
        QString id;
        QString name;
        std::vector<ave::ManagedModel> models;
    };

    std::vector<FamilyBucket> families;
    std::unordered_map<std::string, std::size_t> familyRows;
    for (const auto& model : modelManager_.allModels()) {
        if (!isCompatibleWithBackend(model, backend)) {
            continue;
        }

        const QString familyId = modelFamilyId(model.entry.id);
        const std::string familyKey = familyId.toStdString();
        auto [it, inserted] = familyRows.emplace(familyKey, families.size());
        if (inserted) {
            families.push_back({
                familyId,
                modelFamilyName(model.entry.id, model.entry.displayName),
                {}
            });
        }
        families[it->second].models.push_back(model);
    }

    const QSignalBlocker comboBlocker(modelFamilyCombo_);
    const QSignalBlocker listBlocker(familyCapabilitiesView_);

    modelFamilyCombo_->clear();
    familyCapabilitiesView_->clear();
    currentFamilyModels_.clear();
    familyDraftStages_.clear();
    activeFamilyCapability_.reset();

    if (families.empty()) {
        modelFamilyCombo_->addItem("No backend-compatible model families available", QString());
        modelFamilyCombo_->setEnabled(false);
        return;
    }

    modelFamilyCombo_->setEnabled(true);

    int selectedFamilyRow = 0;
    for (std::size_t i = 0; i < families.size(); ++i) {
        QStringList offeredStages;
        std::unordered_set<int> seenKinds;
        for (const auto& model : families[i].models) {
            const int kindValue = static_cast<int>(model.entry.stage);
            if (seenKinds.insert(kindValue).second) {
                offeredStages << stageTitle(model.entry.stage);
            }
        }

        QString label = families[i].name;
        if (!offeredStages.isEmpty()) {
            label += " - " + joinStageTitles(offeredStages);
        }
        modelFamilyCombo_->addItem(label, families[i].id);
        if (!requestedFamilyId.isEmpty() && families[i].id == requestedFamilyId) {
            selectedFamilyRow = static_cast<int>(i);
        }
    }
    modelFamilyCombo_->setCurrentIndex(selectedFamilyRow);
    currentFamilyModels_ = families[static_cast<std::size_t>(selectedFamilyRow)].models;

    int selectedCapabilityRow = -1;
    int firstReadyRow = -1;
    int firstSelectableRow = -1;
    int row = 0;
    std::unordered_set<int> seenKinds;
    for (const auto& model : currentFamilyModels_) {
        const int kindValue = static_cast<int>(model.entry.stage);
        if (!seenKinds.insert(kindValue).second) {
            continue;
        }

        const bool ready = isReadyForBackend(model, backend);
        const bool needsCompile = needsMigraphxCompile(model, backend);
        const QString status = ready
            ? QStringLiteral("Ready for the selected backend.")
            : (needsCompile
                ? QStringLiteral("Compatible, but needs compilation before it can be added.")
                : QStringLiteral("Not ready for the selected backend."));

        auto previousDraftIt = previousDrafts.find(model.entry.stage);
        if (previousDraftIt != previousDrafts.end()) {
            const auto previousModelIt = previousDraftIt->second.params.find("model");
            const auto previousModelId = previousModelIt != previousDraftIt->second.params.end()
                ? std::get_if<std::string>(&previousModelIt->second)
                : nullptr;
            if (previousModelId != nullptr && *previousModelId == model.entry.id) {
                familyDraftStages_[model.entry.stage] = previousDraftIt->second;
            }
        }
        if (familyDraftStages_.find(model.entry.stage) == familyDraftStages_.end()) {
            familyDraftStages_[model.entry.stage] = defaultDraftStage(model.entry.stage, model.entry.id);
        }

        QString label = stageTitle(model.entry.stage);
        if (ready) {
            label += QStringLiteral("  Ready");
        } else if (needsCompile) {
            label += QStringLiteral("  Needs Compile");
        } else {
            label += QStringLiteral("  Unavailable");
        }

        auto* item = new QListWidgetItem(label, familyCapabilitiesView_);
        item->setData(kRoleModelId, QString::fromStdString(model.entry.id));
        item->setData(kRoleStageKind, kindValue);
        item->setData(kRoleAddEnabled, ready);
        item->setData(kRoleNeedsCompile, needsCompile);
        item->setData(kRoleStatusMessage, status);
        item->setToolTip(QString::fromStdString(model.entry.displayName) + "\n" + status);

        Qt::ItemFlags flags = Qt::ItemIsEnabled | Qt::ItemIsSelectable;
        if (ready) {
            flags |= Qt::ItemIsUserCheckable;
            item->setCheckState(Qt::Unchecked);
            if (firstReadyRow < 0) {
                firstReadyRow = row;
            }
        }
        item->setFlags(flags);

        if (firstSelectableRow < 0) {
            firstSelectableRow = row;
        }
        if (requestedCapability.has_value() && *requestedCapability == model.entry.stage) {
            selectedCapabilityRow = row;
        }
        ++row;
    }

    int preferredCheckedRow = -1;
    if (selectedCapabilityRow >= 0) {
        auto* selectedItem = familyCapabilitiesView_->item(selectedCapabilityRow);
        if (selectedItem != nullptr && selectedItem->data(kRoleAddEnabled).toBool()) {
            preferredCheckedRow = selectedCapabilityRow;
        }
    }
    if (preferredCheckedRow < 0) {
        preferredCheckedRow = firstReadyRow;
    }
    if (preferredCheckedRow >= 0) {
        auto* checkedItem = familyCapabilitiesView_->item(preferredCheckedRow);
        if (checkedItem != nullptr && (checkedItem->flags() & Qt::ItemIsUserCheckable)) {
            checkedItem->setCheckState(Qt::Checked);
        }
    }

    if (selectedCapabilityRow < 0) {
        selectedCapabilityRow = firstReadyRow >= 0 ? firstReadyRow : firstSelectableRow;
    }
    if (selectedCapabilityRow >= 0) {
        familyCapabilitiesView_->setCurrentRow(selectedCapabilityRow);
    }
}

void MainWindow::refreshParamPanel() {
    if (!paramStack_ || !capabilityEditorLabel_ || !extraParamsEdit_) {
        return;
    }

    const auto maybeKind = selectedFamilyCapabilityKind();
    if (!maybeKind.has_value()) {
        activeFamilyCapability_.reset();
        paramStack_->setCurrentIndex(5);
        capabilityEditorLabel_->setText(modelFamilyCombo_ != nullptr && modelFamilyCombo_->isEnabled()
            ? QStringLiteral("Select an enhancement on the left to adjust its parameters.")
            : QStringLiteral("Open Model Manager or switch backends to make a compatible model family available."));
        const QSignalBlocker extraBlocker(extraParamsEdit_);
        extraParamsEdit_->clear();
        extraParamsEdit_->setEnabled(false);
        return;
    }

    const auto maybeModelId = selectedFamilyCapabilityModelId();
    if (!maybeModelId.has_value()) {
        activeFamilyCapability_.reset();
        paramStack_->setCurrentIndex(5);
        capabilityEditorLabel_->setText(QStringLiteral("Select an enhancement on the left to adjust its parameters."));
        const QSignalBlocker extraBlocker(extraParamsEdit_);
        extraParamsEdit_->clear();
        extraParamsEdit_->setEnabled(false);
        return;
    }

    auto draftIt = familyDraftStages_.find(*maybeKind);
    if (draftIt == familyDraftStages_.end()) {
        familyDraftStages_[*maybeKind] = defaultDraftStage(*maybeKind, *maybeModelId);
        draftIt = familyDraftStages_.find(*maybeKind);
    }

    activeFamilyCapability_ = *maybeKind;
    loadStageDraftIntoEditor(draftIt->second);

    QString displayName = QString::fromStdString(*maybeModelId);
    for (const auto& model : currentFamilyModels_) {
        if (model.entry.id == *maybeModelId) {
            displayName = QString::fromStdString(model.entry.displayName);
            break;
        }
    }

    QString help = QStringLiteral("Adjust %1 for %2.")
        .arg(stageTitle(*maybeKind),
             modelFamilyName(*maybeModelId, displayName.toStdString()));
    auto* currentItem = familyCapabilitiesView_->currentItem();
    if (currentItem != nullptr && !currentItem->data(kRoleAddEnabled).toBool()) {
        help += currentItem->data(kRoleNeedsCompile).toBool()
            ? QStringLiteral(" This enhancement needs compilation before it can be added.")
            : QStringLiteral(" This enhancement is not ready for the selected backend.");
    }
    capabilityEditorLabel_->setText(help);
    extraParamsEdit_->setEnabled(true);
}

void MainWindow::refreshStageBuilderActions() {
    if (!familyCapabilitiesView_ || !addStageButton_ || !compileModelButton_ || !modelStatusLabel_) {
        return;
    }

    bool anyCheckedReady = false;
    bool anyNeedsCompile = false;
    QStringList offered;
    QStringList ready;
    QStringList compileNeeded;
    QStringList unavailable;

    for (int row = 0; row < familyCapabilitiesView_->count(); ++row) {
        auto* item = familyCapabilitiesView_->item(row);
        if (item == nullptr) {
            continue;
        }

        const auto kind = static_cast<ave::StageKind>(item->data(kRoleStageKind).toInt());
        const QString title = stageTitle(kind);
        offered << title;
        if (item->data(kRoleAddEnabled).toBool()) {
            ready << title;
            if ((item->flags() & Qt::ItemIsUserCheckable) && item->checkState() == Qt::Checked) {
                anyCheckedReady = true;
            }
        } else if (item->data(kRoleNeedsCompile).toBool()) {
            compileNeeded << title;
            anyNeedsCompile = true;
        } else {
            unavailable << title;
        }
    }

    addStageButton_->setEnabled(anyCheckedReady && !isRunning_.load());
    compileModelButton_->setVisible(anyNeedsCompile);
    compileModelButton_->setEnabled(anyNeedsCompile && !isRunning_.load());

    QStringList lines;
    if (offered.isEmpty()) {
        lines << QStringLiteral("Download or compile a compatible model family to start building stages.");
    } else {
        lines << QStringLiteral("Family offers: %1.").arg(joinStageTitles(offered));
        lines << QStringLiteral("Ready: %1.").arg(ready.isEmpty() ? QStringLiteral("none") : joinStageTitles(ready));
        if (!compileNeeded.isEmpty()) {
            lines << QStringLiteral("Needs compile: %1.").arg(joinStageTitles(compileNeeded));
        }
        if (!unavailable.isEmpty()) {
            lines << QStringLiteral("Unavailable: %1.").arg(joinStageTitles(unavailable));
        }
        if (auto* currentItem = familyCapabilitiesView_->currentItem()) {
            lines << currentItem->data(kRoleStatusMessage).toString();
        }
    }
    modelStatusLabel_->setText(lines.join('\n'));
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

void MainWindow::refreshActiveFilters() {
    if (!activeFiltersView_ || !filterBrowser_) {
        return;
    }

    activeFiltersView_->clear();
    const auto filters = filterBrowser_->activeFilters();
    if (filters.empty()) {
        activeFiltersView_->addItem("No catalog filters enabled.");
        return;
    }

    for (const auto& filter : filters) {
        activeFiltersView_->addItem(
            QString::fromStdString(ave::displayNameForFilter(filter)));
    }
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

    args << "--codec"  << codecCombo_->currentText().trimmed();
    const QString prof = profileCombo_->currentText().trimmed();
    if (!prof.isEmpty() && prof != "(auto)") { args << "--profile" << prof; }
    args << "--crf"    << QString::number(crfSpin_->value());
    args << "--preset" << presetCombo_->currentText().trimmed();

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
    QString family;
    const auto modelIt = stage.params.find("model");
    if (modelIt != stage.params.end()) {
        if (const auto* modelId = std::get_if<std::string>(&modelIt->second)) {
            family = QStringLiteral(" [%1]").arg(modelFamilyName(*modelId, *modelId));
        }
    }

    std::vector<std::string> keys;
    keys.reserve(stage.params.size());
    for (const auto& [k, _] : stage.params) {
        if (k != "model") {
            keys.push_back(k);
        }
    }
    std::sort(keys.begin(), keys.end());

    QStringList parts;
    for (const auto& k : keys) {
        auto it = stage.params.find(k);
        if (it != stage.params.end())
            parts << toQString(k) + "=" + toQString(ave::parameterValueToString(it->second));
    }

    QString base = stageTitle(stage.kind) + family;
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

void MainWindow::storeSelectedFamilyCapabilityDraft() {
    if (!activeFamilyCapability_.has_value()) {
        return;
    }

    std::string modelId;
    auto existingDraft = familyDraftStages_.find(*activeFamilyCapability_);
    if (existingDraft != familyDraftStages_.end()) {
        const auto modelIt = existingDraft->second.params.find("model");
        if (modelIt != existingDraft->second.params.end()) {
            if (const auto* value = std::get_if<std::string>(&modelIt->second)) {
                modelId = *value;
            }
        }
    }
    if (modelId.empty()) {
        for (const auto& model : currentFamilyModels_) {
            if (model.entry.stage == *activeFamilyCapability_) {
                modelId = model.entry.id;
                break;
            }
        }
    }
    if (modelId.empty()) {
        return;
    }

    familyDraftStages_[*activeFamilyCapability_] = captureEditorStage(*activeFamilyCapability_, modelId);
}

std::optional<ave::StageKind> MainWindow::selectedFamilyCapabilityKind() const {
    if (!familyCapabilitiesView_) {
        return std::nullopt;
    }
    auto* item = familyCapabilitiesView_->currentItem();
    if (item == nullptr) {
        return std::nullopt;
    }
    return static_cast<ave::StageKind>(item->data(kRoleStageKind).toInt());
}

std::optional<std::string> MainWindow::selectedFamilyCapabilityModelId() const {
    if (!familyCapabilitiesView_) {
        return std::nullopt;
    }
    auto* item = familyCapabilitiesView_->currentItem();
    if (item == nullptr) {
        return std::nullopt;
    }
    const QString modelId = item->data(kRoleModelId).toString().trimmed();
    if (modelId.isEmpty()) {
        return std::nullopt;
    }
    return modelId.toStdString();
}

ave::EnhancementStage MainWindow::defaultDraftStage(const ave::StageKind kind,
                                                    const std::string& modelId) const {
    ave::EnhancementStage stage;
    stage.kind = kind;
    stage.params["model"] = modelId;

    switch (kind) {
        case ave::StageKind::RestoreCompression:
        case ave::StageKind::RemoveArtifacts:
        case ave::StageKind::Denoise:
        case ave::StageKind::Deblur:
        case ave::StageKind::Dehalo:
            stage.params["strength"] = 0.8;
            break;
        case ave::StageKind::ColorFix:
            stage.params["contrast"] = 1.0;
            stage.params["brightness"] = 1.0;
            stage.params["saturation"] = 1.0;
            stage.params["gamma"] = 1.0;
            stage.params["vibrance"] = 1.0;
            break;
        case ave::StageKind::Upscale:
            stage.params["width"] = std::int64_t{3840};
            stage.params["height"] = std::int64_t{2160};
            break;
        case ave::StageKind::Sharpen:
            stage.params["amount"] = 0.5;
            break;
        case ave::StageKind::Interpolate:
            stage.params["fps"] = std::int64_t{60};
            stage.params["scene_detect"] = true;
            break;
    }
    return stage;
}

ave::EnhancementStage MainWindow::captureEditorStage(const ave::StageKind kind,
                                                     const std::string& modelId) const {
    ave::EnhancementStage stage;
    stage.kind = kind;

    const int page = paramPageForKind(stage.kind);
    if (page == 0) {
        stage.params["strength"] = strengthSlider_->value() / 100.0;
    } else if (page == 1) {
        stage.params["contrast"]   = contrastSlider_->value()   / 100.0;
        stage.params["brightness"] = brightnessSlider_->value() / 100.0;
        stage.params["saturation"] = saturationSlider_->value() / 100.0;
        stage.params["gamma"]      = gammaSlider_->value()      / 100.0;
        stage.params["vibrance"]   = vibranceSlider_->value()   / 100.0;
    } else if (page == 2) {
        stage.params["width"]  = static_cast<std::int64_t>(upscaleWidthSpin_->value());
        stage.params["height"] = static_cast<std::int64_t>(upscaleHeightSpin_->value());
    } else if (page == 3) {
        stage.params["amount"] = sharpenSlider_->value() / 100.0;
    } else if (page == 4) {
        stage.params["fps"] = static_cast<std::int64_t>(interpolateFpsSpin_->value());
        stage.params["scene_detect"] = sceneDetectToggle_->isChecked();
    }

    const QString extra = extraParamsEdit_->text().trimmed();
    if (!extra.isEmpty()) {
        for (const QString& pair : extra.split(',', Qt::SkipEmptyParts)) {
            qsizetype eq = pair.indexOf('=');
            if (eq <= 0) continue;
            const QString key = pair.left(static_cast<int>(eq)).trimmed();
            const QString val = pair.mid(static_cast<int>(eq) + 1).trimmed();
            if (!key.isEmpty() && key.compare(QStringLiteral("model"), Qt::CaseInsensitive) != 0) {
                stage.params[key.toStdString()] = parseParameterValue(val);
            }
        }
    }
    stage.params["model"] = modelId;
    return stage;
}

void MainWindow::loadStageDraftIntoEditor(const ave::EnhancementStage& stage) {
    paramStack_->setCurrentIndex(paramPageForKind(stage.kind));

    auto doubleParam = [&stage](const std::string& key, double fallback) {
        double value = fallback;
        if (ave::tryGetDouble(stage.params, key, value)) {
            return value;
        }
        return fallback;
    };
    auto intParam = [&stage](const std::string& key, const std::int64_t fallback) {
        std::int64_t value = fallback;
        if (ave::tryGetInt(stage.params, key, value)) {
            return value;
        }
        return fallback;
    };
    auto boolParam = [&stage](const std::string& key, const bool fallback) {
        const auto it = stage.params.find(key);
        if (it == stage.params.end()) {
            return fallback;
        }
        if (const auto* value = std::get_if<bool>(&it->second)) {
            return *value;
        }
        if (const auto* value = std::get_if<std::int64_t>(&it->second)) {
            return *value != 0;
        }
        if (const auto* value = std::get_if<double>(&it->second)) {
            return *value != 0.0;
        }
        if (const auto* value = std::get_if<std::string>(&it->second)) {
            if (QString::fromStdString(*value).compare(QStringLiteral("true"), Qt::CaseInsensitive) == 0) {
                return true;
            }
            if (QString::fromStdString(*value).compare(QStringLiteral("false"), Qt::CaseInsensitive) == 0) {
                return false;
            }
        }
        return fallback;
    };

    {
        const QSignalBlocker blocker(strengthSlider_);
        const int sliderValue = std::clamp(static_cast<int>(doubleParam("strength", 0.8) * 100.0), 0, 200);
        strengthSlider_->setValue(sliderValue);
        strengthLabel_->setText(sliderPct(sliderValue));
    }
    {
        const QSignalBlocker contrastBlocker(contrastSlider_);
        const QSignalBlocker brightnessBlocker(brightnessSlider_);
        const QSignalBlocker saturationBlocker(saturationSlider_);
        const QSignalBlocker gammaBlocker(gammaSlider_);
        const QSignalBlocker vibranceBlocker(vibranceSlider_);
        const int contrastValue = std::clamp(static_cast<int>(doubleParam("contrast", 1.0) * 100.0), 0, 200);
        const int brightnessValue = std::clamp(static_cast<int>(doubleParam("brightness", 1.0) * 100.0), 0, 200);
        const int saturationValue = std::clamp(static_cast<int>(doubleParam("saturation", 1.0) * 100.0), 0, 200);
        const int gammaValue = std::clamp(static_cast<int>(doubleParam("gamma", 1.0) * 100.0), 0, 200);
        const int vibranceValue = std::clamp(static_cast<int>(doubleParam("vibrance", 1.0) * 100.0), 0, 200);
        contrastSlider_->setValue(contrastValue);
        brightnessSlider_->setValue(brightnessValue);
        saturationSlider_->setValue(saturationValue);
        gammaSlider_->setValue(gammaValue);
        vibranceSlider_->setValue(vibranceValue);
        contrastLabel_->setText(sliderPct(contrastValue));
        brightnessLabel_->setText(sliderPct(brightnessValue));
        saturationLabel_->setText(sliderPct(saturationValue));
        gammaLabel_->setText(sliderPct(gammaValue));
        vibranceLabel_->setText(sliderPct(vibranceValue));
    }
    {
        const QSignalBlocker widthBlocker(upscaleWidthSpin_);
        const QSignalBlocker heightBlocker(upscaleHeightSpin_);
        upscaleWidthSpin_->setValue(std::clamp(static_cast<int>(intParam("width", 3840)), 16, 16384));
        upscaleHeightSpin_->setValue(std::clamp(static_cast<int>(intParam("height", 2160)), 16, 16384));
    }
    {
        const QSignalBlocker blocker(sharpenSlider_);
        const int sliderValue = std::clamp(static_cast<int>(doubleParam("amount", 0.5) * 100.0), 0, 200);
        sharpenSlider_->setValue(sliderValue);
        sharpenLabel_->setText(sliderPct(sliderValue));
    }
    {
        const QSignalBlocker fpsBlocker(interpolateFpsSpin_);
        const QSignalBlocker sceneBlocker(sceneDetectToggle_);
        interpolateFpsSpin_->setValue(std::clamp(static_cast<int>(intParam("fps", 60)), 1, 240));
        sceneDetectToggle_->setChecked(boolParam("scene_detect", true));
    }

    std::unordered_set<std::string> builtInKeys = {"model"};
    switch (stage.kind) {
        case ave::StageKind::RestoreCompression:
        case ave::StageKind::RemoveArtifacts:
        case ave::StageKind::Denoise:
        case ave::StageKind::Deblur:
        case ave::StageKind::Dehalo:
            builtInKeys.insert("strength");
            break;
        case ave::StageKind::ColorFix:
            builtInKeys.insert("contrast");
            builtInKeys.insert("brightness");
            builtInKeys.insert("saturation");
            builtInKeys.insert("gamma");
            builtInKeys.insert("vibrance");
            break;
        case ave::StageKind::Upscale:
            builtInKeys.insert("width");
            builtInKeys.insert("height");
            break;
        case ave::StageKind::Sharpen:
            builtInKeys.insert("amount");
            break;
        case ave::StageKind::Interpolate:
            builtInKeys.insert("fps");
            builtInKeys.insert("scene_detect");
            break;
    }

    std::vector<std::string> extraKeys;
    extraKeys.reserve(stage.params.size());
    for (const auto& [key, _] : stage.params) {
        if (builtInKeys.find(key) == builtInKeys.end()) {
            extraKeys.push_back(key);
        }
    }
    std::sort(extraKeys.begin(), extraKeys.end());

    QStringList extraPairs;
    for (const auto& key : extraKeys) {
        const auto it = stage.params.find(key);
        if (it != stage.params.end()) {
            extraPairs << toQString(key) + "=" + toQString(ave::parameterValueToString(it->second));
        }
    }

    const QSignalBlocker extraBlocker(extraParamsEdit_);
    extraParamsEdit_->setText(extraPairs.join(", "));
}

// ─────────────────────────────────────────────────────────────────
// Pipeline manipulation
// ─────────────────────────────────────────────────────────────────

void MainWindow::addStage() {
    storeSelectedFamilyCapabilityDraft();

    bool addedAny = false;
    for (int row = 0; row < familyCapabilitiesView_->count(); ++row) {
        auto* item = familyCapabilitiesView_->item(row);
        if (item == nullptr || !item->data(kRoleAddEnabled).toBool()) {
            continue;
        }
        if (!(item->flags() & Qt::ItemIsUserCheckable) || item->checkState() != Qt::Checked) {
            continue;
        }

        const auto kind = static_cast<ave::StageKind>(item->data(kRoleStageKind).toInt());
        auto draftIt = familyDraftStages_.find(kind);
        if (draftIt != familyDraftStages_.end()) {
            stages_.push_back(draftIt->second);
        } else {
            const QString modelId = item->data(kRoleModelId).toString();
            stages_.push_back(defaultDraftStage(kind, modelId.toStdString()));
        }
        addedAny = true;
    }

    if (!addedAny) {
        QMessageBox::information(this,
                                 tr("No Enhancements Selected"),
                                 tr("Check at least one ready enhancement in the selected family before adding it to the pipeline."));
        return;
    }

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
    if (stages_.empty()) {
        return;
    }
    if (appSettings_.confirmBeforeClearPipeline) {
        const QString prompt = "Remove all " +
            QString::number(static_cast<int>(stages_.size())) +
            " requested stages from the pipeline?";
        if (QMessageBox::question(this, "Clear Pipeline", prompt) != QMessageBox::Yes) {
            return;
        }
    }
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
    job.encode.codec    = codecCombo_->currentText().trimmed().toStdString();
    {
        const QString prof = profileCombo_->currentText().trimmed();
        job.encode.profile = (prof == "(auto)") ? "" : prof.toStdString();
    }
    job.encode.crf      = crfSpin_->value();
    job.encode.preset   = presetCombo_->currentText().trimmed().toStdString();
    job.encode.threads  = appSettings_.ffmpegThreads;
    job.dryRun          = dryRunToggle_->isChecked();
    job.previewFrameInterval = appSettings_.previewFrameInterval;
    job.catalogFilters  = filterBrowser_->activeFilters();

    if (!job.dryRun && job.inputPath.empty())  { error = "Input path is required.";  return std::nullopt; }
    if (!job.dryRun && job.outputPath.empty()) { error = "Output path is required."; return std::nullopt; }
    if (job.encode.codec.empty())              { error = "Codec is required.";        return std::nullopt; }
    if (job.encode.preset.empty())             { error = "Preset is required.";       return std::nullopt; }
    return job;
}

void MainWindow::setRunning(bool running) {
    isRunning_.store(running);
    runButton_->setEnabled(!running);
    previewButton_->setEnabled(!running);

    // Show pause/cancel only while running
    pauseButton_->setVisible(running);
    pauseButton_->setEnabled(running);
    pauseButton_->setText("⏸  Pause");
    cancelButton_->setVisible(running);
    cancelButton_->setEnabled(running);

    // Reset flags when starting a new run
    if (running) {
        cancelFlag_.store(false);
        pauseFlag_.store(false);
    }

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
    refreshStageBuilderActions();
}

void MainWindow::appendLog(const QString& line) {
    const QString ts = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    logView_->appendPlainText("[" + ts + "] " + line);
    if (appSettings_.autoScrollLog) {
        logView_->moveCursor(QTextCursor::End);
    }
}

void MainWindow::applySettingsToUi(const bool restoreRememberedPaths) {
    if (!backendCombo_ || !codecCombo_ || !profileCombo_ || !presetCombo_ ||
        !crfSpin_ || !dryRunToggle_ || !previewDurationSpin_) {
        return;
    }

    const int backendIndex = backendCombo_->findData(
        static_cast<int>(appSettings_.defaultBackend));
    if (backendIndex >= 0) {
        backendCombo_->setCurrentIndex(backendIndex);
    }

    const QString codec = QString::fromStdString(appSettings_.defaultCodec);
    if (!codec.trimmed().isEmpty()) {
        codecCombo_->setCurrentText(codec);
    }

    const QString profile = QString::fromStdString(appSettings_.defaultProfile).trimmed();
    if (profile.isEmpty()) {
        profileCombo_->setCurrentIndex(0);
    } else {
        profileCombo_->setCurrentText(profile);
    }

    crfSpin_->setValue(appSettings_.defaultCrf);

    const QString preset = QString::fromStdString(appSettings_.defaultPreset).trimmed();
    if (!preset.isEmpty()) {
        presetCombo_->setCurrentText(preset);
    }

    dryRunToggle_->setChecked(appSettings_.defaultDryRun);
    previewDurationSpin_->setValue(std::max(1, appSettings_.defaultPreviewDurationSec));

    if (restoreRememberedPaths && appSettings_.rememberLastPaths) {
        const QString lastInput = QString::fromStdString(appSettings_.lastInputPath).trimmed();
        const QString lastOutput = QString::fromStdString(appSettings_.lastOutputPath).trimmed();
        if (!lastInput.isEmpty()) {
            inputPathEdit_->setText(lastInput);
        }
        if (!lastOutput.isEmpty()) {
            setManagedOutputPath(lastOutput, false);
        } else if (!lastInput.isEmpty()) {
            applySuggestedOutputPath(lastInput, true);
        }
    } else if (!inputPathEdit_->text().trimmed().isEmpty() && outputPathAutoManaged_) {
        applySuggestedOutputPath(inputPathEdit_->text(), true);
    }
}

void MainWindow::persistUiStateToSettings() {
    if (!inputPathEdit_ || !outputPathEdit_) {
        return;
    }
    if (appSettings_.rememberLastPaths) {
        appSettings_.lastInputPath = inputPathEdit_->text().trimmed().toStdString();
        appSettings_.lastOutputPath = outputPathEdit_->text().trimmed().toStdString();
    } else {
        appSettings_.lastInputPath.clear();
        appSettings_.lastOutputPath.clear();
    }
    appSettings_.save();
}

QString MainWindow::suggestedOutputPathForInput(const QString& inputPath,
                                                const QString& suffixOverride) const {
    const QString trimmed = inputPath.trimmed();
    if (trimmed.isEmpty()) {
        return {};
    }

    const QFileInfo fi(trimmed);
    QString suffix = suffixOverride.trimmed();
    if (suffix.isEmpty()) {
        suffix = QString::fromStdString(appSettings_.outputSuffix).trimmed();
    }
    if (suffix.isEmpty()) {
        suffix = "_enhanced";
    }

    QString extension = fi.suffix();
    if (extension.isEmpty()) {
        extension = "mp4";
    }

    return fi.path() + "/output/" + fi.completeBaseName() + suffix + "." + extension;
}

void MainWindow::applySuggestedOutputPath(const QString& inputPath, const bool force) {
    if (!appSettings_.autoFillOutputPath) {
        return;
    }
    if (!force && !outputPathAutoManaged_ && !outputPathEdit_->text().trimmed().isEmpty()) {
        return;
    }

    const QString suggested = suggestedOutputPathForInput(inputPath);
    if (!suggested.isEmpty()) {
        setManagedOutputPath(suggested, true);
    }
}

void MainWindow::setManagedOutputPath(const QString& path, const bool autoManaged) {
    updatingOutputPath_ = true;
    outputPathEdit_->setText(path);
    updatingOutputPath_ = false;
    outputPathAutoManaged_ = autoManaged;
}

void MainWindow::runJob() {
    if (isRunning_.load()) return;
    QString err;
    auto maybeJob = buildJob(err);
    if (!maybeJob) { QMessageBox::warning(this, "Invalid Configuration", err); return; }

    if (appSettings_.confirmBeforeRun) {
        const QString modeLabel = maybeJob->dryRun ? "dry-run plan" : "processing job";
        const QString prompt = "Start " + modeLabel + " for:\n" +
            toQString(maybeJob->outputPath) + "\n\nRequested stages: " +
            QString::number(static_cast<int>(maybeJob->requestedStages.size()));
        if (QMessageBox::question(this, "Confirm Run", prompt) != QMessageBox::Yes) {
            return;
        }
    }

    appendLog("Job started.");
    setRunning(true);

    ave::VideoJob job = *maybeJob;
    job.cancelFlag = &cancelFlag_;
    job.pauseFlag  = &pauseFlag_;
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

    // Wire live frame preview callback for full runs too
    job.framePreviewCb = [self](const std::uint8_t* rgb, int width, int height) {
        const int byteCount = width * height * 3;
        QByteArray data(reinterpret_cast<const char*>(rgb), byteCount);
        QMetaObject::invokeMethod(self, [self, data, width, height]() {
            if (!self) return;
            QImage img(reinterpret_cast<const uchar*>(data.constData()),
                       width, height, width * 3, QImage::Format_RGB888);
            QPixmap pm = QPixmap::fromImage(img);
            self->framePreviewLabel_->setPixmap(
                pm.scaled(self->framePreviewLabel_->size(),
                          Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }, Qt::QueuedConnection);
    };

    std::thread([self, job]() {
        ave::VideoProcessor processor;
        std::string error;
        const bool ok = processor.process(job, error);
        if (!self) return;
        const bool wasCancelled = job.cancelFlag && job.cancelFlag->load();
        const QString resultLine = wasCancelled
            ? "Processing cancelled by user."
            : (ok ? "Job completed successfully."
                  : "Job failed: " + toQString(error));
        QMetaObject::invokeMethod(self, [self, ok, wasCancelled, resultLine]() {
            if (!self) return;
            self->appendLog(resultLine);
            self->setRunning(false);
            if (!ok && !wasCancelled)
                QMessageBox::critical(self, "Processing Failed", resultLine);
        }, Qt::QueuedConnection);
    }).detach();
}

void MainWindow::runPreview() {
    if (isRunning_.load()) return;
    QString err;
    auto maybeJob = buildJob(err);
    if (!maybeJob) { QMessageBox::warning(this, "Invalid Configuration", err); return; }

    appendLog("Preview started (" + QString::number(previewDurationSpin_->value()) + " seconds).");
    setRunning(true);

    // Reset preview display
    framePreviewLabel_->setText("Processing preview\u2026");

    ave::VideoJob job = *maybeJob;
    job.cancelFlag = &cancelFlag_;
    job.pauseFlag  = &pauseFlag_;
    job.previewMode = true;
    job.previewDurationSec = static_cast<double>(previewDurationSpin_->value());
    job.previewFrameInterval = std::max(1, appSettings_.previewFrameInterval);

    // Auto-generate preview output path
    {
        QFileInfo fi(QString::fromStdString(job.outputPath));
        job.outputPath = (fi.path() + "/" + fi.completeBaseName()
                          + "_preview." + fi.suffix()).toStdString();
    }

    QPointer<MainWindow> self(this);

    // Wire progress callback
    job.progressCb = [self](int overallPct, int taskPct, const std::string& msg) {
        const QString qmsg = QString::fromStdString(msg);
        QMetaObject::invokeMethod(self, [self, overallPct, taskPct, qmsg]() {
            if (!self) return;
            self->progressBar_->setValue(overallPct);
            self->taskProgressBar_->setValue(taskPct);
            if (!qmsg.isEmpty()) {
                self->progressLabel_->setText(qmsg);
                if (qmsg != self->lastProgressMsg_) {
                    self->lastProgressMsg_ = qmsg;
                    self->appendLog(qmsg);
                }
            }
        }, Qt::QueuedConnection);
    };

    // Wire live frame preview callback
    job.framePreviewCb = [self](const std::uint8_t* rgb, int width, int height) {
        // Copy frame data — it's only valid during this call
        const int byteCount = width * height * 3;
        QByteArray data(reinterpret_cast<const char*>(rgb), byteCount);
        QMetaObject::invokeMethod(self, [self, data, width, height]() {
            if (!self) return;
            QImage img(reinterpret_cast<const uchar*>(data.constData()),
                       width, height, width * 3, QImage::Format_RGB888);
            QPixmap pm = QPixmap::fromImage(img);
            self->framePreviewLabel_->setPixmap(
                pm.scaled(self->framePreviewLabel_->size(),
                          Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }, Qt::QueuedConnection);
    };

    std::thread([self, job]() {
        ave::VideoProcessor processor;
        std::string error;
        const bool ok = processor.process(job, error);
        if (!self) return;
        const bool wasCancelled = job.cancelFlag && job.cancelFlag->load();
        const QString resultLine = wasCancelled
            ? "Preview cancelled by user."
            : (ok ? "Preview completed — output: " + toQString(job.outputPath)
                  : "Preview failed: " + toQString(error));
        QMetaObject::invokeMethod(self, [self, ok, wasCancelled, resultLine]() {
            if (!self) return;
            self->appendLog(resultLine);
            self->setRunning(false);
            if (!ok && !wasCancelled)
                QMessageBox::critical(self, "Preview Failed", resultLine);
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
    ModelManagerDialog dlg(modelManager_, appSettings_, QString(), this);
    dlg.exec();
    modelManager_.refresh();
    refreshModelFamilies();
    refreshParamPanel();
    refreshStageBuilderActions();
}

void MainWindow::compileSelectedModel() {
    QString modelId;
    if (auto* currentItem = familyCapabilitiesView_->currentItem();
        currentItem != nullptr && currentItem->data(kRoleNeedsCompile).toBool()) {
        modelId = currentItem->data(kRoleModelId).toString();
    }
    if (modelId.isEmpty()) {
        for (int row = 0; row < familyCapabilitiesView_->count(); ++row) {
            auto* item = familyCapabilitiesView_->item(row);
            if (item != nullptr && item->data(kRoleNeedsCompile).toBool()) {
                modelId = item->data(kRoleModelId).toString();
                break;
            }
        }
    }
    if (modelId.isEmpty()) {
        if (const auto selectedId = selectedFamilyCapabilityModelId(); selectedId.has_value()) {
            modelId = QString::fromStdString(*selectedId);
        }
    }
    if (modelId.isEmpty()) {
        QMessageBox::information(this, tr("No Model Selected"),
                                 tr("Select a model family first."));
        return;
    }

    ModelManagerDialog dlg(modelManager_, appSettings_, modelId, this);
    dlg.exec();
    modelManager_.refresh();
    refreshModelFamilies();
    refreshParamPanel();
    refreshStageBuilderActions();
}

void MainWindow::openSettings() {
    SettingsDialog dlg(appSettings_, this);
    connect(&dlg, &SettingsDialog::settingsApplied, this, [this]() {
        applySettingsToUi(false);
        refreshCommandPreview();
    });
    dlg.exec();
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
    root.insert("codec",   codecCombo_->currentText().trimmed());
    {
        const QString prof = profileCombo_->currentText().trimmed();
        if (!prof.isEmpty() && prof != "(auto)") root.insert("profile", prof);
    }
    root.insert("crf",    crfSpin_->value());
    root.insert("preset", presetCombo_->currentText().trimmed());
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
    codecCombo_->setCurrentText(root.value("codec").toString("libx264"));
    {
        const QString prof = root.value("profile").toString();
        if (!prof.isEmpty()) profileCombo_->setCurrentText(prof);
        else profileCombo_->setCurrentIndex(0);  // "(auto)"
    }
    presetCombo_->setCurrentText(root.value("preset").toString("medium"));
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
    if (index == 1) setManagedOutputPath(suggestedOutputPathForInput(inputPathEdit_->text(), "_cleanup_60fps"), true);
    if (index == 2) setManagedOutputPath(suggestedOutputPathForInput(inputPathEdit_->text(), "_anime_4k_60fps"), true);
    if (index == 3) setManagedOutputPath(suggestedOutputPathForInput(inputPathEdit_->text(), "_archive_restore_1440p"), true);
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
        applySuggestedOutputPath(path, false);
        appendLog("Input loaded by drag-and-drop: " + path);
    }
    refreshCommandPreview();
    event->acceptProposedAction();
}
