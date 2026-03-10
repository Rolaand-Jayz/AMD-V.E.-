#pragma once

#include <unordered_map>
#include <vector>

#include <QWidget>

#include "ave/filter_catalog.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QLabel;
class QScrollArea;
class QSlider;
class QVBoxLayout;

// ─────────────────────────────────────────────────────────────────
// FilterParamRow — one parameter: label + slider + spinbox
// ─────────────────────────────────────────────────────────────────
struct FilterParamRow {
    std::string      key;
    double           defaultValue = 0.0;
    QLabel*          label   = nullptr;
    QSlider*         slider  = nullptr;
    QDoubleSpinBox*  spin    = nullptr;
};

// ─────────────────────────────────────────────────────────────────
// FilterRow — one filter card in the browser
// ─────────────────────────────────────────────────────────────────
struct FilterRow {
    std::string                filterId;
    QWidget*                   card       = nullptr;
    QCheckBox*                 enableBox  = nullptr;
    QGroupBox*                 paramGroup = nullptr;
    std::vector<FilterParamRow> paramRows;
};

// ─────────────────────────────────────────────────────────────────
// FilterBrowser — scrollable panel listing every embedded filter
// ─────────────────────────────────────────────────────────────────
// Groups filters by category.  Each filter has:
//   • Checkbox to enable/disable
//   • Collapsible parameter section with sliders & spinboxes
//   • Category/Runtime badge
//
// Toggling any control emits filtersChanged().
//
// getActiveFilters() returns the full list for backend consumption.
// ─────────────────────────────────────────────────────────────────
class FilterBrowser final : public QWidget {
    Q_OBJECT

  public:
    explicit FilterBrowser(QWidget* parent = nullptr);
    ~FilterBrowser() override = default;

    /// Return all enabled filters with their current parameter values.
    std::vector<ave::ActiveFilter> activeFilters() const;

    /// Enable a filter by id (programmatic).
    void setFilterEnabled(const std::string& id, bool enabled);

    /// Set a filter's parameter value (programmatic).
    void setFilterParam(const std::string& id,
                        const std::string& key, double value);

    /// Reset all filters to their default disabled state.
    void clearAllFilters();

    /// Replace the active filter set in one pass.
    void setActiveFilters(const std::vector<ave::ActiveFilter>& filters);

    /// Get count of enabled filters.
    int enabledCount() const;

  signals:
    /// Emitted whenever any filter is toggled or any param changes.
    void filtersChanged();

  private:
    void buildUi();

    /// Filter by category dropdown changes.
    void onCategoryFilterChanged(int index);

    QComboBox*     categoryFilter_ = nullptr;
    QScrollArea*   scrollArea_     = nullptr;
    QVBoxLayout*   filterLayout_   = nullptr;

    std::vector<FilterRow> rows_;
};
