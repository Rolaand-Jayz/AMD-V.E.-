#pragma once

#include <memory>

#include <QDialog>

#include "ave/app_settings.hpp"
#include "ave/model_manager.hpp"

class QLabel;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QPushButton;
class QSplitter;
class QTextEdit;

// ─────────────────────────────────────────────────────────────────
// ModelManagerDialog
// ─────────────────────────────────────────────────────────────────
// Provides a UI for:
//  • Browsing all models grouped by stage kind
//  • Seeing each model's download / compile status
//  • Downloading models (with progress)
//  • Compiling downloaded ONNX models to MiGraphX .mxr
//  • Cancelling in-flight operations
// ─────────────────────────────────────────────────────────────────
class ModelManagerDialog final : public QDialog {
    Q_OBJECT

  public:
    explicit ModelManagerDialog(ave::ModelManager&  manager,
                                ave::AppSettings&   settings,
                                const QString&      initialModelId = QString(),
                                QWidget*            parent = nullptr);
    ~ModelManagerDialog() override = default;

  private slots:
    void onSelectionChanged();
    void onDownloadClicked();
    void onConvertClicked();
    void onCancelClicked();
    void onOpenFolderClicked();
    void onRefreshClicked();

  private:
    // Prevent the dialog from being closed while a download or
    // conversion is still running.  Background threads hold raw 'this'
    // pointers through callbacks and would dereference a dangling
    // pointer if the dialog were destroyed mid-flight.
    void closeEvent(QCloseEvent* event) override;

    void buildUi();
    void populateList();
    void updateDetailPanel(const QString& modelId);
    void setButtonsEnabled(bool enabled);

    // Build per-call callbacks that marshal back to the Qt main thread.
    ave::ModelProgressCb makeProgressCb(const std::string& modelId);
    ave::ModelStateCb    makeStateCb   (const std::string& modelId);

    // Called on the Qt main thread by the callbacks above.
    void onProgressQt   (const QString& modelId, float progress, const QString& msg);
    void onStateChangedQt(const QString& modelId, ave::ModelState newState);

    ave::ModelManager&  manager_;
    ave::AppSettings&   settings_;

    QListWidget*  modelList_   = nullptr;
    QLabel*       nameLabel_   = nullptr;
    QLabel*       stageLabel_  = nullptr;
    QLabel*       statusLabel_ = nullptr;
    QLabel*       pathLabel_   = nullptr;
    QLabel*       descLabel_   = nullptr;
    QProgressBar* progressBar_ = nullptr;
    QLabel*       progressMsg_ = nullptr;

    QPushButton* downloadBtn_  = nullptr;
    QPushButton* convertBtn_   = nullptr;
    QPushButton* cancelBtn_    = nullptr;
    QPushButton* openFolderBtn_= nullptr;
    QPushButton* refreshBtn_   = nullptr;
    QPushButton* closeBtn_     = nullptr;

    QString selectedModelId_;
    bool operationKickoff_ = false;
};
