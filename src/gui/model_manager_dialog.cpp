#include "model_manager_dialog.hpp"

#include <thread>

#include <QApplication>
#include <QComboBox>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSplitter>
#include <QString>
#include <QUrl>
#include <QVBoxLayout>

#include "ave/model_catalog.hpp"
#include "ave/types.hpp"

using namespace ave;

// ─── Helpers ─────────────────────────────────────────────────────

static QString stateText(ModelState s) {
    switch (s) {
    case ModelState::NotDownloaded: return QStringLiteral("Not downloaded");
    case ModelState::Downloading:   return QStringLiteral("Downloading…");
    case ModelState::Downloaded:    return QStringLiteral("Downloaded");
    case ModelState::Converting:    return QStringLiteral("Converting to MiGraphX…");
    case ModelState::Converted:     return QStringLiteral("Converted (.mxr)");
    case ModelState::Optimizing:    return QStringLiteral("Optimising for hardware…");
    case ModelState::Optimized:     return QStringLiteral("Optimised");
    case ModelState::Error:         return QStringLiteral("Error");
    }
    return QStringLiteral("Unknown");
}

static QString stateColor(ModelState s) {
    switch (s) {
    case ModelState::NotDownloaded: return QStringLiteral("#9E9E9E");
    case ModelState::Downloading:
    case ModelState::Converting:
    case ModelState::Optimizing:    return QStringLiteral("#2196F3");
    case ModelState::Downloaded:    return QStringLiteral("#FF9800");
    case ModelState::Converted:     return QStringLiteral("#4CAF50");
    case ModelState::Optimized:     return QStringLiteral("#00BCD4");
    case ModelState::Error:         return QStringLiteral("#F44336");
    }
    return QStringLiteral("#9E9E9E");
}

// Build per-call callbacks that post safely back to the UI thread.
// QMetaObject::invokeMethod with a QObject* context and
// Qt::QueuedConnection will automatically drop the call if the
// context object is destroyed before the event is processed.
ModelProgressCb ModelManagerDialog::makeProgressCb(const std::string& modelId) {
    return [this, modelId](const std::string&, float p, const std::string& msg) {
        QString qid  = QString::fromStdString(modelId);
        QString qmsg = QString::fromStdString(msg);
        QMetaObject::invokeMethod(this,
            [this, qid, p, qmsg]() { onProgressQt(qid, p, qmsg); },
            Qt::QueuedConnection);
    };
}

ModelStateCb ModelManagerDialog::makeStateCb(const std::string& modelId) {
    return [this, modelId](const std::string&, ModelState st) {
        QString qid = QString::fromStdString(modelId);
        QMetaObject::invokeMethod(this,
            [this, qid, st]() { onStateChangedQt(qid, st); },
            Qt::QueuedConnection);
    };
}

// ─── Construction ────────────────────────────────────────────────

ModelManagerDialog::ModelManagerDialog(ModelManager& manager, AppSettings& settings, QWidget* parent)
    : QDialog(parent), manager_(manager), settings_(settings) {
    setWindowTitle(tr("Model Manager"));
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    resize(880, 560);
    buildUi();
    populateList();
}

// ─── UI construction ─────────────────────────────────────────────

void ModelManagerDialog::buildUi() {
    auto* root = new QHBoxLayout(this);
    root->setContentsMargins(12, 12, 12, 12);
    root->setSpacing(8);

    auto* leftPane   = new QWidget;
    auto* leftLayout = new QVBoxLayout(leftPane);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    modelList_ = new QListWidget;
    modelList_->setMinimumWidth(280);
    modelList_->setSelectionMode(QAbstractItemView::SingleSelection);
    leftLayout->addWidget(new QLabel(tr("Models")));
    leftLayout->addWidget(modelList_, 1);

    refreshBtn_ = new QPushButton(tr("Refresh"));
    refreshBtn_->setToolTip(tr("Re-scan model directories"));
    leftLayout->addWidget(refreshBtn_);

    auto* rightPane   = new QWidget;
    auto* rightLayout = new QVBoxLayout(rightPane);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->setSpacing(6);

    nameLabel_   = new QLabel(tr("Select a model"));
    nameLabel_->setStyleSheet(QStringLiteral("font-size: 14pt; font-weight: bold;"));
    stageLabel_  = new QLabel;
    stageLabel_->setStyleSheet(QStringLiteral("color: #777;"));
    statusLabel_ = new QLabel;
    pathLabel_   = new QLabel;
    pathLabel_->setWordWrap(true);
    descLabel_   = new QLabel;
    descLabel_->setWordWrap(true);
    descLabel_->setStyleSheet(QStringLiteral("color: #555;"));

    // Per-model precision override combo
    precisionCombo_ = new QComboBox;
    precisionCombo_->addItem(tr("Global default"),  QStringLiteral("global"));
    precisionCombo_->addItem(tr("fp32"),            QStringLiteral("fp32"));
    precisionCombo_->addItem(tr("fp16"),            QStringLiteral("fp16"));
    precisionCombo_->addItem(tr("int8"),            QStringLiteral("int8"));
    precisionCombo_->setEnabled(false);
    precisionCombo_->setToolTip(tr("Per-model compile precision. "
        "\"Global default\" follows the setting in Settings > Inference."));

    rightLayout->addWidget(nameLabel_);
    rightLayout->addWidget(stageLabel_);
    rightLayout->addWidget(descLabel_);
    rightLayout->addSpacing(4);
    rightLayout->addWidget(new QLabel(tr("Status:")));
    rightLayout->addWidget(statusLabel_);
    rightLayout->addWidget(new QLabel(tr("Path:")));
    rightLayout->addWidget(pathLabel_);
    {
        auto* precRow = new QHBoxLayout;
        precRow->addWidget(new QLabel(tr("Compile precision:")));
        precRow->addWidget(precisionCombo_, 1);
        rightLayout->addLayout(precRow);
    }

    progressBar_ = new QProgressBar;
    progressBar_->setRange(0, 100);
    progressBar_->setVisible(false);
    progressMsg_ = new QLabel;
    progressMsg_->setVisible(false);
    rightLayout->addWidget(progressBar_);
    rightLayout->addWidget(progressMsg_);
    rightLayout->addStretch();

    auto* btnLayout = new QHBoxLayout;
    downloadBtn_   = new QPushButton(tr("Download"));
    convertBtn_    = new QPushButton(tr("Convert to MiGraphX"));
    optimizeBtn_   = new QPushButton(tr("Optimise for Hardware"));
    cancelBtn_     = new QPushButton(tr("Cancel"));
    openFolderBtn_ = new QPushButton(tr("Open Folder"));
    cancelBtn_->setEnabled(false);
    btnLayout->addWidget(downloadBtn_);
    btnLayout->addWidget(convertBtn_);
    btnLayout->addWidget(optimizeBtn_);
    btnLayout->addWidget(cancelBtn_);
    btnLayout->addWidget(openFolderBtn_);
    btnLayout->addStretch();
    rightLayout->addLayout(btnLayout);

    auto* bottomRow = new QHBoxLayout;
    bottomRow->addStretch();
    closeBtn_ = new QPushButton(tr("Close"));
    closeBtn_->setDefault(true);
    bottomRow->addWidget(closeBtn_);
    rightLayout->addLayout(bottomRow);

    auto* splitter = new QSplitter(Qt::Horizontal);
    splitter->addWidget(leftPane);
    splitter->addWidget(rightPane);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    root->addWidget(splitter);

    connect(modelList_,    &QListWidget::currentItemChanged,
            this, &ModelManagerDialog::onSelectionChanged);
    connect(downloadBtn_,  &QPushButton::clicked, this, &ModelManagerDialog::onDownloadClicked);
    connect(convertBtn_,   &QPushButton::clicked, this, &ModelManagerDialog::onConvertClicked);
    connect(optimizeBtn_,  &QPushButton::clicked, this, &ModelManagerDialog::onOptimizeClicked);
    connect(cancelBtn_,    &QPushButton::clicked, this, &ModelManagerDialog::onCancelClicked);
    connect(openFolderBtn_,&QPushButton::clicked, this, &ModelManagerDialog::onOpenFolderClicked);
    connect(refreshBtn_,   &QPushButton::clicked, this, &ModelManagerDialog::onRefreshClicked);
    connect(closeBtn_,     &QPushButton::clicked, this, &QDialog::accept);

    // Precision combo: save override when user changes it.
    connect(precisionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int) {
        if (selectedModelId_.isEmpty()) return;
        const QString key = precisionCombo_->currentData().toString();
        const std::string id = selectedModelId_.toStdString();
        if (key == QStringLiteral("global")) {
            settings_.modelPrecisionOverrides.erase(id);
        } else {
            settings_.modelPrecisionOverrides[id] = precisionFromString(key.toStdString()).value_or(ModelPrecision::Fp32);
        }
        settings_.save();
    });

    setButtonsEnabled(false);
}

// ─── List population ─────────────────────────────────────────────

void ModelManagerDialog::populateList() {
    const QString currentId = selectedModelId_;
    modelList_->clear();
    manager_.refresh();

    const StageKind order[] = {
        StageKind::RestoreCompression, StageKind::RemoveArtifacts,
        StageKind::Denoise, StageKind::Deblur, StageKind::Dehalo,
        StageKind::ColorFix, StageKind::Upscale,
        StageKind::Sharpen, StageKind::Interpolate,
    };

    for (StageKind sk : order) {
        bool headerAdded = false;
        for (const auto& m : manager_.allModels()) {
            if (m.entry.stage != sk) continue;
            if (!headerAdded) {
                auto* header = new QListWidgetItem(
                    QStringLiteral("── ") + QString::fromStdString(toString(sk)) + QStringLiteral(" ──"));
                header->setFlags(Qt::ItemIsEnabled);
                QFont f = header->font(); f.setBold(true); header->setFont(f);
                header->setForeground(QColor(QStringLiteral("#2196F3")));
                header->setData(Qt::UserRole, QString());
                modelList_->addItem(header);
                headerAdded = true;
            }
            auto* item = new QListWidgetItem(
                QStringLiteral("  ") + QString::fromStdString(m.entry.displayName));
            item->setData(Qt::UserRole, QString::fromStdString(m.entry.id));
            item->setForeground(QColor(stateColor(m.state)));
            item->setToolTip(stateText(m.state));
            modelList_->addItem(item);
            if (QString::fromStdString(m.entry.id) == currentId)
                modelList_->setCurrentItem(item);
        }
    }
}

// ─── Detail panel ────────────────────────────────────────────────

void ModelManagerDialog::updateDetailPanel(const QString& modelId) {
    selectedModelId_ = modelId;

    if (modelId.isEmpty()) {
        nameLabel_->setText(tr("Select a model"));
        stageLabel_->setText({}); statusLabel_->setText({});
        pathLabel_->setText({}); descLabel_->setText({});
        progressBar_->setVisible(false); progressMsg_->setVisible(false);
        setButtonsEnabled(false);
        return;
    }

    auto optModel = manager_.findModel(modelId.toStdString());
    if (!optModel) return;
    const ManagedModel& m = *optModel;

    nameLabel_->setText(QString::fromStdString(m.entry.displayName));
    stageLabel_->setText(tr("Stage: %1").arg(QString::fromStdString(toString(m.entry.stage))));
    descLabel_->setText(QString::fromStdString(m.entry.description));
    statusLabel_->setText(
        QStringLiteral("<span style='color:%1;font-weight:bold;'>%2</span>")
            .arg(stateColor(m.state), stateText(m.state)));

    auto bestPath = manager_.bestPathForModel(modelId.toStdString());
    pathLabel_->setText(bestPath ? QString::fromStdString(*bestPath) : tr("(not on disk)"));

    updatePrecisionCombo(m);

    const bool busy = (m.state == ModelState::Downloading ||
                       m.state == ModelState::Converting  ||
                       m.state == ModelState::Optimizing);
    progressBar_->setVisible(busy);
    progressMsg_->setVisible(busy);
    if (!busy) progressBar_->setValue(0);

    setButtonsEnabled(true);
    downloadBtn_->setEnabled(!m.entry.downloadUrl.empty() && m.state == ModelState::NotDownloaded);
    const bool canConvert =
        m.state == ModelState::Downloaded &&
        (m.entry.sourceFormat == ModelFormat::Onnx || m.entry.sourceFormat == ModelFormat::Pytorch);
    convertBtn_->setEnabled(canConvert);
    optimizeBtn_->setEnabled(m.state == ModelState::Converted);
    cancelBtn_->setEnabled(busy);
    openFolderBtn_->setEnabled(true);
    precisionCombo_->setEnabled(!busy);
}

// ─── Precision helpers ───────────────────────────────────────────

void ModelManagerDialog::updatePrecisionCombo(const ManagedModel& m) {
    // Block signals so our currentIndexChanged handler doesn't fire
    // while we are programmatically updating the combo.
    QSignalBlocker blocker(precisionCombo_);

    // Update the "Global default" label to show the current global value.
    const QString globalLabel = tr("Global default (%1)")
        .arg(QString::fromStdString(toString(settings_.globalQuantization)));
    precisionCombo_->setItemText(0, globalLabel);

    // Select the per-model override if one exists, else "Global default".
    const std::string id = m.entry.id;
    auto it = settings_.modelPrecisionOverrides.find(id);
    if (it != settings_.modelPrecisionOverrides.end()) {
        const QString overrideStr = QString::fromStdString(toString(it->second));
        const int idx = precisionCombo_->findData(overrideStr);
        precisionCombo_->setCurrentIndex(idx >= 0 ? idx : 0);
    } else {
        precisionCombo_->setCurrentIndex(0);  // "Global default"
    }
}

ave::ModelPrecision ModelManagerDialog::effectivePrecision() const {
    if (selectedModelId_.isEmpty()) return ModelPrecision::Fp32;

    auto optModel = manager_.findModel(selectedModelId_.toStdString());
    const ModelPrecision catalogPrec = optModel ? optModel->entry.precision : ModelPrecision::Fp32;

    const QString key = precisionCombo_->currentData().toString();
    if (key != QStringLiteral("global")) {
        // Use the explicit per-model override from the combo.
        const ModelPrecision requested = precisionFromString(key.toStdString()).value_or(ModelPrecision::Fp32);
        return AppSettings::clampToEnvironment(requested, true /* MiGraphX required anyway */);
    }
    // Fall back to global-aware logic (honours catalogue when global is still default Fp32).
    const ModelPrecision eff = settings_.effectivePrecisionFor(selectedModelId_.toStdString(), catalogPrec);
    return AppSettings::clampToEnvironment(eff, true);
}

void ModelManagerDialog::setButtonsEnabled(bool enabled) {
    downloadBtn_->setEnabled(enabled);
    convertBtn_->setEnabled(enabled);
    optimizeBtn_->setEnabled(enabled);
    cancelBtn_->setEnabled(false);
    openFolderBtn_->setEnabled(enabled);
}

// ─── Qt-thread callbacks ──────────────────────────────────────────

void ModelManagerDialog::onProgressQt(const QString& modelId, float p, const QString& msg) {
    if (modelId != selectedModelId_) return;
    progressBar_->setVisible(true); progressMsg_->setVisible(true);
    progressBar_->setValue(static_cast<int>(p * 100.f));
    progressMsg_->setText(msg);
}

void ModelManagerDialog::onStateChangedQt(const QString& modelId, ModelState) {
    for (int i = 0; i < modelList_->count(); ++i) {
        QListWidgetItem* it = modelList_->item(i);
        if (it->data(Qt::UserRole).toString() == modelId) {
            auto opt = manager_.findModel(modelId.toStdString());
            if (opt) { it->setForeground(QColor(stateColor(opt->state))); it->setToolTip(stateText(opt->state)); }
            break;
        }
    }
    if (modelId == selectedModelId_) updateDetailPanel(selectedModelId_);
}

// ─── Slots ───────────────────────────────────────────────────────

void ModelManagerDialog::onSelectionChanged() {
    auto* item = modelList_->currentItem();
    if (!item) { updateDetailPanel({}); return; }
    updateDetailPanel(item->data(Qt::UserRole).toString());
}

void ModelManagerDialog::onDownloadClicked() {
    if (selectedModelId_.isEmpty()) return;
    const std::string id = selectedModelId_.toStdString();
    std::thread([this, id]() {
        std::string err;
        manager_.startDownload(id, makeProgressCb(id), makeStateCb(id), err);
    }).detach();
    updateDetailPanel(selectedModelId_);
}

void ModelManagerDialog::onConvertClicked() {
    if (selectedModelId_.isEmpty()) return;
    const std::string id    = selectedModelId_.toStdString();
    const ModelPrecision prec = effectivePrecision();
    std::thread([this, id, prec]() {
        std::string err;
        if (!manager_.convertToMiGraphX(id, makeProgressCb(id), makeStateCb(id), err, prec)) {
            const QString msg = QString::fromStdString("MiGraphX conversion failed:\n" + err);
            QMetaObject::invokeMethod(QApplication::instance(), [msg]() {
                QMessageBox::warning(nullptr, "Conversion Failed", msg);
            }, Qt::QueuedConnection);
        }
    }).detach();
    updateDetailPanel(selectedModelId_);
}

void ModelManagerDialog::onOptimizeClicked() {
    if (selectedModelId_.isEmpty()) return;
    const std::string id    = selectedModelId_.toStdString();
    const ModelPrecision prec = effectivePrecision();
    std::thread([this, id, prec]() {
        std::string err;
        if (!manager_.optimizeForHardware(id, makeProgressCb(id), makeStateCb(id), err, prec)) {
            const QString msg = QString::fromStdString("Hardware optimisation failed:\n" + err);
            QMetaObject::invokeMethod(QApplication::instance(), [msg]() {
                QMessageBox::warning(nullptr, "Optimisation Failed", msg);
            }, Qt::QueuedConnection);
        }
    }).detach();
    updateDetailPanel(selectedModelId_);
}

void ModelManagerDialog::onCancelClicked() {
    if (selectedModelId_.isEmpty()) return;
    manager_.cancelDownload(selectedModelId_.toStdString());
    updateDetailPanel(selectedModelId_);
}

void ModelManagerDialog::onOpenFolderClicked() {
    QDesktopServices::openUrl(QUrl::fromLocalFile(
        QString::fromStdString(manager_.modelsDirectory())));
}

void ModelManagerDialog::onRefreshClicked() {
    populateList();
    if (!selectedModelId_.isEmpty()) updateDetailPanel(selectedModelId_);
}
