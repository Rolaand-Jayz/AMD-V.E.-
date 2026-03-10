#pragma once

#include <QDialog>

#include "ave/app_settings.hpp"

class QCheckBox;
class QComboBox;
class QLabel;
class QListWidget;
class QLineEdit;
class QSpinBox;
class QStackedWidget;

class SettingsDialog final : public QDialog {
    Q_OBJECT

  public:
    explicit SettingsDialog(ave::AppSettings& settings,
                            QWidget* parent = nullptr);
    ~SettingsDialog() override = default;

  signals:
    void settingsApplied();

  private slots:
    void onApplyClicked();
    void onResetDefaults();
    void onReloadDiagnostics();
    void onOpenConfigDirectory();
    void onOpenModelsDirectory();

  private:
    void buildUi();
    void loadFromSettings();
    void loadWidgetsFromSettings(const ave::AppSettings& settings);
    void applyToSettings();
    void refreshDiagnostics();
    QString modelsDirectoryPath() const;
    static QString configDirectoryPath();

    ave::AppSettings& settings_;

    QListWidget*  navList_              = nullptr;
    QStackedWidget* pageStack_          = nullptr;

    QLabel* settingsPathLabel_          = nullptr;
    QLabel* modelsPathLabel_            = nullptr;
    QLabel* rememberedPathsLabel_       = nullptr;
    QLabel* backendSummaryLabel_        = nullptr;
    QLabel* toolchainSummaryLabel_      = nullptr;

    QComboBox* defaultBackendCombo_     = nullptr;
    QComboBox* codecCombo_              = nullptr;
    QLineEdit* profileEdit_             = nullptr;
    QSpinBox*  crfSpin_                 = nullptr;
    QLineEdit* presetEdit_              = nullptr;
    QSpinBox*  threadsSpin_             = nullptr;

    QCheckBox* defaultDryRunCheck_      = nullptr;
    QSpinBox*  previewDurationSpin_     = nullptr;
    QSpinBox*  previewFrameIntervalSpin_ = nullptr;
    QCheckBox* autoFillOutputCheck_     = nullptr;
    QLineEdit* outputSuffixEdit_        = nullptr;
    QCheckBox* rememberPathsCheck_      = nullptr;
    QCheckBox* confirmRunCheck_         = nullptr;
    QCheckBox* confirmClearPipelineCheck_ = nullptr;

    QCheckBox* autoScrollLogCheck_      = nullptr;
    QCheckBox* verboseLogCheck_         = nullptr;
};
