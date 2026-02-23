#pragma once

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include <QMainWindow>

#include "ave/app_settings.hpp"
#include "ave/backend_manager.hpp"
#include "ave/job.hpp"
#include "ave/model_manager.hpp"
#include "ave/planner.hpp"

class QComboBox;
class QDoubleSpinBox;
class QDragEnterEvent;
class QDropEvent;
class QGroupBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSlider;
class QSpinBox;
class QStackedWidget;
class ToggleSwitch;

// ─────────────────────────────────────────────────────────────────
// MainWindow
// ─────────────────────────────────────────────────────────────────
// Primary application window.  Provides:
//   • Input / output selection (drag-and-drop supported)
//   • Backend + encode settings (with ToggleSwitch for dry-run)
//   • Stage builder:  kind dropdown + model dropdown + per-stage
//     parameter sliders that adapt to the selected stage kind
//   • Reorderable pipeline list (drag in list + move up/down buttons)
//   • Planned execution order preview
//   • Quick templates
//   • Profile save/load (JSON)
//   • Command-line preview
//   • Session log
//   • "Model Manager" button opening ModelManagerDialog
// ─────────────────────────────────────────────────────────────────
class MainWindow final : public QMainWindow {
    Q_OBJECT

  public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override = default;

  protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;

  private:
    // ── UI construction ─────────────────────────────────────────
    void buildUi();
    void wireActions();

    // ── Stage parameter panels (one per StageKind) ───────────────
    QWidget* buildStrengthPanel(QWidget* parent);   // generic strength
    QWidget* buildUpscalePanel(QWidget* parent);
    QWidget* buildSharpenPanel(QWidget* parent);
    QWidget* buildInterpolatePanel(QWidget* parent);
    QWidget* buildColorFixPanel(QWidget* parent);
    QWidget* buildEmptyPanel(QWidget* parent);

    // ── Refresh helpers ──────────────────────────────────────────
    void refreshModelCombo();        // populate model dropdown for current kind
    void refreshParamPanel();        // switch stacked widget page
    void refreshRequestedStages();
    void refreshPlannedStages();
    void refreshCommandPreview();

    // ── Stage operations ─────────────────────────────────────────
    void addStage();
    void removeSelectedStage();
    void moveSelectedStage(int delta);
    void clearStages();
    void onPipelineSelectionChanged();

    // ── Job ──────────────────────────────────────────────────────
    void runJob();
    void probeBackends();
    void openModelManager();
    void openSettings();

    // ── Profiles ─────────────────────────────────────────────────
    void saveProfile();
    void loadProfile();
    void applyQuickTemplate(int index);

    // ── Utilities ────────────────────────────────────────────────
    void appendLog(const QString& line);
    void setRunning(bool running);
    std::optional<ave::VideoJob> buildJob(QString& error) const;
    ave::EnhancementStage buildStageFromEditor() const;
    static QString stageToDisplay(const ave::EnhancementStage& stage);
    static QString stageToCommandSpec(const ave::EnhancementStage& stage);

    // ── Settings ─────────────────────────────────────────────────
    ave::AppSettings appSettings_;

    // ── Model manager ────────────────────────────────────────────
    ave::ModelManager modelManager_;

    // ── I/O ──────────────────────────────────────────────────────
    QLineEdit* inputPathEdit_  = nullptr;
    QLineEdit* outputPathEdit_ = nullptr;

    // ── Encode settings ──────────────────────────────────────────
    QComboBox*    backendCombo_ = nullptr;
    QLineEdit*    codecEdit_    = nullptr;
    QSpinBox*     crfSpin_      = nullptr;
    QLineEdit*    presetEdit_   = nullptr;
    ToggleSwitch* dryRunToggle_ = nullptr;

    // ── Stage builder ────────────────────────────────────────────
    QComboBox*     stageKindCombo_  = nullptr;
    QComboBox*     modelCombo_      = nullptr;   // model for current kind
    QStackedWidget* paramStack_     = nullptr;   // one page per StageKind

    // Strength panel widgets
    QSlider*       strengthSlider_  = nullptr;
    QLabel*        strengthLabel_   = nullptr;

    // Upscale panel widgets
    QSpinBox*      upscaleWidthSpin_  = nullptr;
    QSpinBox*      upscaleHeightSpin_ = nullptr;

    // Sharpen panel widgets
    QSlider*  sharpenSlider_ = nullptr;
    QLabel*   sharpenLabel_  = nullptr;

    // Interpolate panel widgets
    QSpinBox*     interpolateFpsSpin_   = nullptr;
    ToggleSwitch* sceneDetectToggle_    = nullptr;

    // ColorFix panel widgets
    QSlider* contrastSlider_   = nullptr;
    QSlider* brightnessSlider_ = nullptr;
    QSlider* saturationSlider_ = nullptr;
    QSlider* gammaSlider_      = nullptr;
    QSlider* vibranceSlider_   = nullptr;
    QLabel*  contrastLabel_    = nullptr;
    QLabel*  brightnessLabel_  = nullptr;
    QLabel*  saturationLabel_  = nullptr;
    QLabel*  gammaLabel_       = nullptr;
    QLabel*  vibranceLabel_    = nullptr;

    // Extra params
    QLineEdit* extraParamsEdit_ = nullptr;

    // ── Pipeline list ─────────────────────────────────────────────
    QListWidget* requestedStagesView_ = nullptr;   // user-order, drag-reorder
    QListWidget* plannedStagesView_   = nullptr;   // planner order

    // ── Progress ──────────────────────────────────────────────
    QProgressBar* progressBar_     = nullptr;  ///< Overall job progress (0–100)
    QProgressBar* taskProgressBar_ = nullptr;  ///< Current task progress (0–100)
    QLabel*       progressLabel_   = nullptr;  ///< Status text
    QString       lastProgressMsg_;            ///< Used to detect log-worthy message changes

    // ── Action buttons ────────────────────────────────────────────
    QPushButton* runButton_         = nullptr;
    QPushButton* removeStageButton_ = nullptr;
    QPushButton* moveUpButton_      = nullptr;
    QPushButton* moveDownButton_    = nullptr;
    QPushButton* clearStagesButton_ = nullptr;

    // ── Utilities bar ─────────────────────────────────────────────
    QComboBox* quickTemplateCombo_  = nullptr;
    QLineEdit* commandPreviewEdit_  = nullptr;
    QPlainTextEdit* logView_        = nullptr;

    // ── State ─────────────────────────────────────────────────────
    std::vector<ave::EnhancementStage> stages_;
    ave::PipelinePlanner planner_;
    std::atomic<bool> isRunning_{false};
};
