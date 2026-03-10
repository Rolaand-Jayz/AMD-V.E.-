#include "model_manager_dialog.hpp"

#include <thread>

#include <QApplication>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QPointer>
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
    case ModelState::Error:         return QStringLiteral("Error");
    }
    return QStringLiteral("Unknown");
}

static QString stateColor(ModelState s) {
    switch (s) {
    case ModelState::NotDownloaded: return QStringLiteral("#9E9E9E");
    case ModelState::Downloading:
    case ModelState::Converting:    return QStringLiteral("#2196F3");
    case ModelState::Downloaded:    return QStringLiteral("#FF9800");
    case ModelState::Converted:     return QStringLiteral("#4CAF50");
    case ModelState::Error:         return QStringLiteral("#F44336");
    }
    return QStringLiteral("#9E9E9E");
}

// Build per-call callbacks that post safely back to the UI thread.
ModelProgressCb ModelManagerDialog::makeProgressCb(const std::string& modelId) {
    QPointer<ModelManagerDialog> self(this);
    return [self, modelId](const std::string&, float p, const std::string& msg) {
        QString qid  = QString::fromStdString(modelId);
        QString qmsg = QString::fromStdString(msg);
        QMetaObject::invokeMethod(QApplication::instance(),
            [self, qid, p, qmsg]() {
                if (!self) { return; }
                self->onProgressQt(qid, p, qmsg);
            }, Qt::QueuedConnection);
    };
}

ModelStateCb ModelManagerDialog::makeStateCb(const std::string& modelId) {
    QPointer<ModelManagerDialog> self(this);
    return [self, modelId](const std::string&, ModelState st) {
        QString qid = QString::fromStdString(modelId);
        QMetaObject::invokeMethod(QApplication::instance(),
            [self, qid, st]() {
                if (!self) { return; }
                self->onStateChangedQt(qid, st);
            }, Qt::QueuedConnection);
    };
}

// ─── Construction ────────────────────────────────────────────────

// ─── closeEvent ──────────────────────────────────────────────────
// Block the dialog from closing while any model operation is in progress.
void ModelManagerDialog::closeEvent(QCloseEvent* event) {
    if (operationKickoff_) {
        QMessageBox::information(
            this,
            tr("Operation in progress"),
            tr("A model operation is starting. Please wait a moment and try again."));
        event->ignore();
        return;
    }

    const auto models = manager_.allModels();
    for (const auto& m : models) {
        if (m.state == ModelState::Downloading ||
            m.state == ModelState::Converting) {
            QMessageBox::information(
                this,
                tr("Operation in progress"),
                tr("Please wait for the current operation to finish before closing.\n\n"
                   "You can cancel an active download with the Cancel button."));
            event->ignore();
            return;
        }
    }
    QDialog::closeEvent(event);
}

ModelManagerDialog::ModelManagerDialog(ModelManager& manager,
                                       AppSettings& settings,
                                       const QString& initialModelId,
                                       QWidget* parent)
    : QDialog(parent), manager_(manager), settings_(settings) {
    setWindowTitle(tr("Model Manager"));
    setWindowFlags(windowFlags() & ~Qt::WindowContextHelpButtonHint);
    resize(880, 560);
    buildUi();
    selectedModelId_ = initialModelId;
    populateList();
    if (!selectedModelId_.isEmpty()) {
        updateDetailPanel(selectedModelId_);
    }
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

    rightLayout->addWidget(nameLabel_);
    rightLayout->addWidget(stageLabel_);
    rightLayout->addWidget(descLabel_);
    rightLayout->addSpacing(4);
    rightLayout->addWidget(new QLabel(tr("Status:")));
    rightLayout->addWidget(statusLabel_);
    rightLayout->addWidget(new QLabel(tr("Path:")));
    rightLayout->addWidget(pathLabel_);

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
    cancelBtn_     = new QPushButton(tr("Cancel"));
    openFolderBtn_ = new QPushButton(tr("Open Folder"));
    cancelBtn_->setEnabled(false);
    btnLayout->addWidget(downloadBtn_);
    btnLayout->addWidget(convertBtn_);
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
    connect(cancelBtn_,    &QPushButton::clicked, this, &ModelManagerDialog::onCancelClicked);
    connect(openFolderBtn_,&QPushButton::clicked, this, &ModelManagerDialog::onOpenFolderClicked);
    connect(refreshBtn_,   &QPushButton::clicked, this, &ModelManagerDialog::onRefreshClicked);
    connect(closeBtn_,     &QPushButton::clicked, this, &QDialog::accept);

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

    const bool busy = (m.state == ModelState::Downloading ||
                       m.state == ModelState::Converting);
    progressBar_->setVisible(busy);
    progressMsg_->setVisible(busy);
    if (!busy) {
        // Restore determinate range if it was in indeterminate (pulsing) mode.
        if (progressBar_->maximum() == 0) { progressBar_->setRange(0, 100); }
        progressBar_->setValue(0);
    }

    setButtonsEnabled(true);
    downloadBtn_->setEnabled(!m.entry.downloadUrl.empty() && m.state == ModelState::NotDownloaded);
    const bool canConvert =
        m.state == ModelState::Downloaded &&
        (m.entry.sourceFormat == ModelFormat::Onnx || m.entry.sourceFormat == ModelFormat::Pytorch);
    convertBtn_->setEnabled(canConvert);
    cancelBtn_->setEnabled(busy);
    openFolderBtn_->setEnabled(true);
}

void ModelManagerDialog::setButtonsEnabled(bool enabled) {
    downloadBtn_->setEnabled(enabled);
    convertBtn_->setEnabled(enabled);
    cancelBtn_->setEnabled(false);
    openFolderBtn_->setEnabled(enabled);
}

// ─── Qt-thread callbacks ──────────────────────────────────────────

void ModelManagerDialog::onProgressQt(const QString& modelId, float p, const QString& msg) {
    if (modelId != selectedModelId_) return;
    progressBar_->setVisible(true); progressMsg_->setVisible(true);
    if (p < 0.0f) {
        // Indeterminate (pulsing) mode — the compilation phase has no
        // measurable sub-progress; only elapsed time is known.
        if (progressBar_->maximum() != 0) { progressBar_->setRange(0, 0); }
    } else {
        // Determinate mode — restore 0-100 range if we were pulsing.
        if (progressBar_->maximum() == 0) { progressBar_->setRange(0, 100); }
        progressBar_->setValue(static_cast<int>(p * 100.f));
    }
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
    operationKickoff_ = false;
    for (const auto& m : manager_.allModels()) {
        if (m.state == ModelState::Downloading || m.state == ModelState::Converting) {
            operationKickoff_ = true;
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
    operationKickoff_ = true;
    std::string err;
    if (!manager_.startDownload(id, makeProgressCb(id), makeStateCb(id), err)) {
        operationKickoff_ = false;
        QMessageBox::warning(this, tr("Download Failed"), QString::fromStdString(err));
    }
    updateDetailPanel(selectedModelId_);
}

void ModelManagerDialog::onConvertClicked() {
    if (selectedModelId_.isEmpty()) return;
    const std::string id = selectedModelId_.toStdString();
    operationKickoff_ = true;
    const auto progressCb = makeProgressCb(id);
    const auto stateCb = makeStateCb(id);
    ModelManager* manager = &manager_;
    QPointer<ModelManagerDialog> self(this);
    std::thread([manager, id, progressCb, stateCb, self]() {
        std::string err;
        if (!manager->convertToMiGraphX(id, progressCb, stateCb, err)) {
            const QString msg = QString::fromStdString("MiGraphX conversion failed:\n" + err);
            QMetaObject::invokeMethod(QApplication::instance(), [msg]() {
                QMessageBox::warning(nullptr, "Conversion Failed", msg);
            }, Qt::QueuedConnection);
            QMetaObject::invokeMethod(QApplication::instance(), [self]() {
                if (!self) { return; }
                self->operationKickoff_ = false;
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
