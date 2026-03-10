#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <QMainWindow>

#include "ave/app_settings.hpp"
#include "ave/backend_manager.hpp"
#include "ave/job.hpp"
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

    // ── Stage operations ─────────────────────────────────────────
    void addStage();
    void removeSelectedStage();
    void moveSelectedStage(int delta);
    void clearStages();
    void onPipelineSelectionChanged();

    // ── Job ──────────────────────────────────────────────────────
    void runJob();
    void runPreview();
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
    void applySettingsToUi(bool restoreRememberedPaths);
    void persistUiStateToSettings();
    QString suggestedOutputPathForInput(const QString& inputPath,
                                        const QString& suffixOverride = QString()) const;
    void applySuggestedOutputPath(const QString& inputPath, bool force);
    void setManagedOutputPath(const QString& path, bool autoManaged);
    void setRunning(bool running);
    std::optional<ave::VideoJob> buildJob(QString& error) const;
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
    QLabel*      filterExecutionSummaryLabel_ = nullptr;
    QLabel*      commandFilterNoteLabel_ = nullptr;

    // ── Progress ──────────────────────────────────────────────
    QProgressBar* progressBar_     = nullptr;  ///< Overall job progress (0–100)
    QProgressBar* taskProgressBar_ = nullptr;  ///< Current task progress (0–100)
    QLabel*       progressLabel_   = nullptr;  ///< Status text
    QString       lastProgressMsg_;            ///< Used to detect log-worthy message changes

    // ── Action buttons ────────────────────────────────────────────
    QPushButton* runButton_         = nullptr;
    QPushButton* previewButton_     = nullptr;   ///< 10-sec preview run
    QPushButton* pauseButton_       = nullptr;   ///< Pause/resume processing
    QPushButton* cancelButton_      = nullptr;   ///< Cancel processing
    QPushButton* addStageButton_    = nullptr;
    QPushButton* compileModelButton_ = nullptr;
    QPushButton* removeStageButton_ = nullptr;
    QPushButton* moveUpButton_      = nullptr;
    QPushButton* moveDownButton_    = nullptr;
    QPushButton* clearStagesButton_ = nullptr;

    // ── Preview ───────────────────────────────────────────────────
    QSpinBox*  previewDurationSpin_ = nullptr;   ///< Preview clip length (seconds)
    QLabel*    framePreviewLabel_   = nullptr;    ///< Live frame display

    // ── Utilities bar ─────────────────────────────────────────────
    QComboBox* quickTemplateCombo_  = nullptr;
    QLineEdit* commandPreviewEdit_  = nullptr;
    QPlainTextEdit* logView_        = nullptr;

    // ── Filter browser ─────────────────────────────────────────
    FilterBrowser* filterBrowser_   = nullptr;

    // ── State ─────────────────────────────────────────────────────
    std::vector<ave::EnhancementStage> stages_;
    std::vector<ave::ManagedModel> currentFamilyModels_;
    std::map<ave::StageKind, ave::EnhancementStage> familyDraftStages_;
    std::optional<ave::StageKind> activeFamilyCapability_;
    ave::PipelinePlanner planner_;
    std::atomic<bool> isRunning_{false};
    std::atomic<bool> cancelFlag_{false};
    std::atomic<bool> pauseFlag_{false};
};
