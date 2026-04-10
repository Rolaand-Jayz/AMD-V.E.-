#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <QMainWindow>

#include "ave/app_settings.hpp"
#include "ave/backend_manager.hpp"
#include "ave/job.hpp"
#include "ave/job_queue.hpp"
#include "ave/job_recovery.hpp"
#include "ave/model_manager.hpp"
#include "ave/planner.hpp"

class FilterBrowser;
class QComboBox;
class QCloseEvent;
class QDoubleSpinBox;
class QDragEnterEvent;
class QDropEvent;
class QGroupBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QPixmap;
class QProgressBar;
class QPushButton;
class QSlider;
class QSpinBox;
class QStackedWidget;
class QTimer;
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
    void closeEvent(QCloseEvent* event) override;
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
    QWidget* buildStereo3dPanel(QWidget* parent);
    QWidget* buildInterpolatePanel(QWidget* parent);
    QWidget* buildColorFixPanel(QWidget* parent);
    QWidget* buildEmptyPanel(QWidget* parent);

    // ── Refresh helpers ──────────────────────────────────────────
    void refreshModelFamilies();
    void refreshParamPanel();        // switch stacked widget page
    void refreshStageBuilderActions();
    void refreshRequestedStages();
    void refreshPlannedStages();
    void refreshActiveFilters();
    void refreshFilterExecutionSummary();
    void refreshCommandPreview();
    void updateFilterPresetDescription();
    void alignManualFilterSelection();
    void applySelectedFilterPreset();
    void clearCatalogFilters();

    // ── Stage operations ─────────────────────────────────────────
    void addStage();
    void removeSelectedStage();
    void moveSelectedStage(int delta);
    void clearStages();
    void onPipelineSelectionChanged();

    // ── Job ──────────────────────────────────────────────────────
    void runJob();
    void runPreview();
    void addCurrentJobToQueue();
    void removeSelectedQueuedJob();
    void clearFinishedQueuedJobs();
    void runQueuedJobs();
    void probeBackends();
    void openModelManager();
    void compileSelectedModel();
    void openSettings();

    // ── Profiles ─────────────────────────────────────────────────
    void saveProfile();
    void loadProfile();
    void applyQuickTemplate(int index);

    // ── Utilities ────────────────────────────────────────────────
    void appendLog(const QString& line);
    void applyProgressUpdate(int overallPct, int taskPct, const QString& msg);
    void applySettingsToUi(bool restoreRememberedPaths);
    void persistUiStateToSettings();
    void updateProcessedPreviewTitle(const QString& runtimeName);
    void setPreviewPixmap(QLabel* label, const QPixmap& pixmap);
    void refreshPreviewPlaceholders();
    void refreshOriginalPreviewFrame();
    void requestTelemetryRefresh();
    void applyTelemetryStatus(const QString& summary, const QString& logLine = QString());
    void promptForRecoveredJob();
    void applyRecoveredJob(const ave::RecoveredJobState& state);
    void startRecoveryTracking(const ave::VideoJob& job);
    void clearRecoveryTracking();
    void loadQueuedJobs();
    void persistQueuedJobs();
    void refreshQueuedJobsView();
    void refreshQueueControls();
    std::optional<std::size_t> nextRunnableQueuedJobIndex() const;
    void executeFullJob(ave::VideoJob job, std::optional<std::size_t> queueIndex = std::nullopt);
    void startQueuedJob(std::size_t index);
    void finishQueuedJob(std::size_t index,
                         const ave::VideoJob& job,
                         bool ok,
                         bool wasCancelled,
                         const QString& error);
    QString suggestedOutputPathForInput(const QString& inputPath,
                                        const QString& suffixOverride = QString()) const;
    void applySuggestedOutputPath(const QString& inputPath, bool force);
    void setManagedOutputPath(const QString& path, bool autoManaged);
    void setRunning(bool running);
    std::optional<ave::VideoJob> buildJob(QString& error) const;
    bool hasQueuedOutputConflict(const QString& outputPath,
                                 std::optional<std::size_t> ignoreIndex = std::nullopt) const;
    void storeSelectedFamilyCapabilityDraft();
    std::optional<ave::StageKind> selectedFamilyCapabilityKind() const;
    std::optional<std::string> selectedFamilyCapabilityModelId() const;
    ave::EnhancementStage defaultDraftStage(ave::StageKind kind,
                                            const std::string& modelId) const;
    ave::EnhancementStage captureEditorStage(ave::StageKind kind,
                                             const std::string& modelId) const;
    void loadStageDraftIntoEditor(const ave::EnhancementStage& stage);
    static QString stageToDisplay(const ave::EnhancementStage& stage);
    static QString stageToCommandSpec(const ave::EnhancementStage& stage);

    // ── Settings ─────────────────────────────────────────────────
    ave::AppSettings appSettings_;
    ave::JobQueueStore jobQueueStore_;
    ave::JobRecoveryStore jobRecoveryStore_;

    // ── Model manager ────────────────────────────────────────────
    ave::ModelManager modelManager_;

    // ── I/O ──────────────────────────────────────────────────────
    QLineEdit* inputPathEdit_  = nullptr;
    QLineEdit* outputPathEdit_ = nullptr;
    bool outputPathAutoManaged_ = true;
    bool updatingOutputPath_ = false;

    // ── Encode settings ──────────────────────────────────────────
    QComboBox*    backendCombo_  = nullptr;
    QComboBox*    codecCombo_    = nullptr;
    QComboBox*    profileCombo_  = nullptr;
    QSpinBox*     crfSpin_       = nullptr;
    QComboBox*    presetCombo_   = nullptr;
    ToggleSwitch* dryRunToggle_  = nullptr;

    // ── Stage builder ────────────────────────────────────────────
    QComboBox*     modelFamilyCombo_ = nullptr;
    QListWidget*   familyCapabilitiesView_ = nullptr;
    QStackedWidget* paramStack_     = nullptr;   // one page per StageKind
    QLabel*        modelStatusLabel_ = nullptr;
    QLabel*        capabilityEditorLabel_ = nullptr;

    // Strength panel widgets
    QSlider*       strengthSlider_  = nullptr;
    QLabel*        strengthLabel_   = nullptr;

    // Upscale panel widgets
    QSpinBox*      upscaleWidthSpin_  = nullptr;
    QSpinBox*      upscaleHeightSpin_ = nullptr;

    // Sharpen panel widgets
    QSlider*  sharpenSlider_ = nullptr;
    QLabel*   sharpenLabel_  = nullptr;

    // Stereo 3D panel widgets
    QComboBox* stereoFormatCombo_ = nullptr;
    QComboBox* stereoSyntheticViewCombo_ = nullptr;
    QComboBox* stereoMapperTypeCombo_ = nullptr;
    QComboBox* stereoPadModeCombo_ = nullptr;
    QComboBox* stereoMethodCombo_ = nullptr;
    QDoubleSpinBox* stereoDivergenceSpin_ = nullptr;
    QDoubleSpinBox* stereoConvergenceSpin_ = nullptr;
    QDoubleSpinBox* stereoForegroundScaleSpin_ = nullptr;
    QDoubleSpinBox* stereoIpdOffsetSpin_ = nullptr;
    QDoubleSpinBox* stereoPadSpin_ = nullptr;
    QDoubleSpinBox* stereoEmaDecaySpin_ = nullptr;
    QSpinBox* stereoDepthResolutionSpin_ = nullptr;
    QSpinBox* stereoEdgeDilateXSpin_ = nullptr;
    QSpinBox* stereoEdgeDilateYSpin_ = nullptr;
    ToggleSwitch* stereoLimitResolutionToggle_ = nullptr;
    ToggleSwitch* stereoMetricDepthToggle_ = nullptr;
    ToggleSwitch* stereoDepthAaToggle_ = nullptr;
    ToggleSwitch* stereoEmaNormalizeToggle_ = nullptr;

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
    QListWidget* activeFiltersView_   = nullptr;   // enabled catalog filters
    QComboBox*   filterPresetCombo_   = nullptr;
    QLabel*      filterPresetDescriptionLabel_ = nullptr;
    QLabel*      filterExecutionSummaryLabel_ = nullptr;
    QLabel*      commandFilterNoteLabel_ = nullptr;

    // ── Progress ──────────────────────────────────────────────
    QProgressBar* progressBar_     = nullptr;  ///< Overall job progress (0–100)
    QProgressBar* taskProgressBar_ = nullptr;  ///< Current task progress (0–100)
    QLabel*       progressLabel_   = nullptr;  ///< Status text
    QString       lastProgressMsg_;            ///< Used to detect log-worthy message changes
    QListWidget*  queuedJobsView_ = nullptr;
    QLabel*       queueSummaryLabel_ = nullptr;

    // ── Action buttons ────────────────────────────────────────────
    QPushButton* runButton_         = nullptr;
    QPushButton* previewButton_     = nullptr;   ///< 10-sec preview run
    QPushButton* pauseButton_       = nullptr;   ///< Pause/resume processing
    QPushButton* cancelButton_      = nullptr;   ///< Cancel processing
    QPushButton* addQueueButton_    = nullptr;
    QPushButton* runQueueButton_    = nullptr;
    QPushButton* removeQueueButton_ = nullptr;
    QPushButton* clearFinishedQueueButton_ = nullptr;
    QPushButton* addStageButton_    = nullptr;
    QPushButton* compileModelButton_ = nullptr;
    QPushButton* removeStageButton_ = nullptr;
    QPushButton* moveUpButton_      = nullptr;
    QPushButton* moveDownButton_    = nullptr;
    QPushButton* clearStagesButton_ = nullptr;

    // ── Preview ───────────────────────────────────────────────────
    QSpinBox*  previewDurationSpin_ = nullptr;   ///< Preview clip length (seconds)
    QLabel*    originalPreviewTitleLabel_ = nullptr;
    QLabel*    processedPreviewTitleLabel_ = nullptr;
    QLabel*    originalFramePreviewLabel_ = nullptr;
    QLabel*    framePreviewLabel_   = nullptr;    ///< Live processed frame display
    QLabel*    telemetryStatusLabel_ = nullptr;
    QTimer*    telemetryTimer_ = nullptr;

    // ── Utilities bar ─────────────────────────────────────────────
    QComboBox* quickTemplateCombo_  = nullptr;
    QLineEdit* commandPreviewEdit_  = nullptr;
    QPlainTextEdit* logView_        = nullptr;

    // ── Filter browser ─────────────────────────────────────────
    FilterBrowser* filterBrowser_   = nullptr;

    // ── State ─────────────────────────────────────────────────────
    std::vector<ave::EnhancementStage> stages_;
    std::vector<ave::QueuedJobRecord> queuedJobs_;
    std::vector<ave::ManagedModel> currentFamilyModels_;
    std::map<ave::StageKind, ave::EnhancementStage> familyDraftStages_;
    std::optional<ave::StageKind> activeFamilyCapability_;
    std::optional<std::size_t> activeQueueIndex_;
    bool updatingFilterPresetUi_ = false;
    bool queueRunActive_ = false;
    ave::PipelinePlanner planner_;
    std::atomic<bool> isRunning_{false};
    std::atomic<bool> cancelFlag_{false};
    std::atomic<bool> pauseFlag_{false};
    std::atomic<std::uint64_t> previewRequestSerial_{0};
    std::atomic<bool> telemetryRequestInFlight_{false};
    QString lastTelemetryLogLine_;
};
