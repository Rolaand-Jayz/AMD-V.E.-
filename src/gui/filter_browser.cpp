// DISABLED: VapourSynth/GLSL/FilterCatalog feature — commented out, not removed.
#if 0  // ── entire file disabled ──────────────────────────────

// ─────────────────────────────────────────────────────────────────
// filter_browser.cpp — GUI panel for toggling / configuring filters
// ─────────────────────────────────────────────────────────────────
#include "filter_browser.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QScrollArea>
#include <QSlider>
#include <QVBoxLayout>

#include "ave/filter_catalog.hpp"

// ═════════════════════════════════════════════════════════════════
//  Helpers
// ═════════════════════════════════════════════════════════════════

namespace {

// Map FilterCategory enum values to display-order indices so the
// combo box "All" (0) + categories (1-N) work smoothly.

struct CategoryInfo {
    ave::FilterCategory cat;
    QString             label;
};

const std::vector<CategoryInfo>& categoryList() {
    static const std::vector<CategoryInfo> list = {
        {ave::FilterCategory::Sharpen,          QStringLiteral("Sharpen")},
        {ave::FilterCategory::Denoise,          QStringLiteral("Denoise")},
        {ave::FilterCategory::Deblur,           QStringLiteral("Deblur")},
        {ave::FilterCategory::Dehalo,           QStringLiteral("Dehalo")},
        {ave::FilterCategory::ColorCorrection,  QStringLiteral("Color Correction")},
        {ave::FilterCategory::Restoration,      QStringLiteral("Restoration")},
        {ave::FilterCategory::Upscale,          QStringLiteral("Upscale")},
        {ave::FilterCategory::LineArt,          QStringLiteral("Line Art")},
        {ave::FilterCategory::Grain,            QStringLiteral("Grain")},
        {ave::FilterCategory::Interpolation,    QStringLiteral("Interpolation")},
        {ave::FilterCategory::Utility,          QStringLiteral("Utility")},
    };
    return list;
}

QString runtimeBadge(ave::FilterRuntime rt) {
    switch (rt) {
        case ave::FilterRuntime::Glsl:        return QStringLiteral("[GLSL]");
        case ave::FilterRuntime::VapourSynth: return QStringLiteral("[VS]");
    }
    return {};
}

}  // namespace

// ═════════════════════════════════════════════════════════════════
//  Construction
// ═════════════════════════════════════════════════════════════════

FilterBrowser::FilterBrowser(QWidget* parent) : QWidget(parent) {
    buildUi();
}

void FilterBrowser::buildUi() {
    auto* topLayout = new QVBoxLayout(this);
    topLayout->setContentsMargins(0, 0, 0, 0);

    // ── Category filter combo ───────────────────────────────────
    auto* filterRow = new QHBoxLayout;
    filterRow->addWidget(new QLabel(QStringLiteral("Category:")));
    categoryFilter_ = new QComboBox;
    categoryFilter_->addItem(QStringLiteral("All Categories"));
    for (const auto& ci : categoryList()) {
        categoryFilter_->addItem(ci.label);
    }
    filterRow->addWidget(categoryFilter_, 1);
    topLayout->addLayout(filterRow);

    // ── Scrollable filter list ──────────────────────────────────
    scrollArea_ = new QScrollArea;
    scrollArea_->setWidgetResizable(true);
    auto* scrollContent = new QWidget;
    filterLayout_ = new QVBoxLayout(scrollContent);
    filterLayout_->setSpacing(4);
    filterLayout_->setContentsMargins(2, 2, 2, 2);

    // Populate from catalog.
    const auto& catalog = ave::allEmbeddedFilters();
    for (const auto& ef : catalog) {
        FilterRow row;
        row.filterId = ef.id;

        // ── Card: checkbox + badges ─────────────────────────────
        auto* card = new QVBoxLayout;

        auto* headerRow = new QHBoxLayout;
        row.enableBox = new QCheckBox(
            runtimeBadge(ef.runtime) + QStringLiteral(" ") +
            QString::fromStdString(ef.name));
        row.enableBox->setToolTip(QString::fromStdString(ef.description));
        headerRow->addWidget(row.enableBox, 1);
        auto* catLabel = new QLabel(
            QString::fromStdString(ave::toString(ef.category)));
        catLabel->setStyleSheet(
            QStringLiteral("color: #888; font-size: 10px;"));
        headerRow->addWidget(catLabel);
        card->addLayout(headerRow);

        // ── Parameter controls ──────────────────────────────────
        row.paramGroup = new QGroupBox;
        row.paramGroup->setFlat(true);
        row.paramGroup->setVisible(false);  // collapsed by default
        auto* paramLayout = new QVBoxLayout(row.paramGroup);
        paramLayout->setContentsMargins(16, 4, 4, 4);
        paramLayout->setSpacing(2);

        for (const auto& pd : ef.params) {
            FilterParamRow pr;
            pr.key = pd.key;

            auto* pRow = new QHBoxLayout;
            pr.label = new QLabel(QString::fromStdString(pd.label) +
                                  QStringLiteral(":"));
            pr.label->setToolTip(QString::fromStdString(pd.description));
            pr.label->setMinimumWidth(100);
            pRow->addWidget(pr.label);

            // Slider (int-range mapped to 0..1000 for float)
            pr.slider = new QSlider(Qt::Horizontal);
            pr.slider->setRange(0, 1000);
            int initTick = static_cast<int>(
                1000.0 * (pd.defVal - pd.minVal) / (pd.maxVal - pd.minVal));
            pr.slider->setValue(initTick);
            pRow->addWidget(pr.slider, 1);

            // SpinBox
            pr.spin = new QDoubleSpinBox;
            pr.spin->setRange(pd.minVal, pd.maxVal);
            pr.spin->setSingleStep(pd.step);
            pr.spin->setDecimals(pd.isInt ? 0 : 2);
            pr.spin->setValue(pd.defVal);
            pRow->addWidget(pr.spin);

            paramLayout->addLayout(pRow);

            // ── Slider <-> SpinBox sync ─────────────────────────
            const double pMin  = pd.minVal;
            const double pMax  = pd.maxVal;

            connect(pr.slider, &QSlider::valueChanged, this,
                    [this, spin = pr.spin, pMin, pMax](int v) {
                        double val = pMin + (pMax - pMin) * v / 1000.0;
                        spin->blockSignals(true);
                        spin->setValue(val);
                        spin->blockSignals(false);
                        emit filtersChanged();
                    });

            connect(pr.spin,
                    QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                    this,
                    [this, slider = pr.slider, pMin, pMax](double val) {
                        int tick = static_cast<int>(
                            1000.0 * (val - pMin) / (pMax - pMin));
                        slider->blockSignals(true);
                        slider->setValue(tick);
                        slider->blockSignals(false);
                        emit filtersChanged();
                    });

            row.paramRows.push_back(pr);
        }

        card->addWidget(row.paramGroup);
        filterLayout_->addLayout(card);

        // ── Show/hide params on toggle ──────────────────────────
        connect(row.enableBox, &QCheckBox::toggled, this,
                [this, paramGroup = row.paramGroup](bool checked) {
                    paramGroup->setVisible(checked);
                    emit filtersChanged();
                });

        rows_.push_back(std::move(row));
    }

    filterLayout_->addStretch(1);
    scrollArea_->setWidget(scrollContent);
    topLayout->addWidget(scrollArea_, 1);

    // ── Wire category filter ────────────────────────────────────
    connect(categoryFilter_,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &FilterBrowser::onCategoryFilterChanged);
}

// ═════════════════════════════════════════════════════════════════
//  Category filter
// ═════════════════════════════════════════════════════════════════

void FilterBrowser::onCategoryFilterChanged(int index) {
    const auto& catalog = ave::allEmbeddedFilters();
    for (std::size_t i = 0; i < rows_.size() && i < catalog.size(); ++i) {
        bool visible = true;
        if (index > 0) {
            auto selectedCat = categoryList()[static_cast<std::size_t>(index - 1)].cat;
            visible = (catalog[i].category == selectedCat);
        }
        // Show/hide by finding parent layouts — simpler: walk enableBox parent
        if (rows_[i].enableBox) {
            // We need to show/hide the whole card.  Since we used addLayout
            // (not addWidget), we wrap each card in a helper QWidget.
            // For simplicity, just show/hide the checkbox + paramGroup:
            rows_[i].enableBox->parentWidget()->setVisible(visible);
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  Public API
// ═════════════════════════════════════════════════════════════════

std::vector<ave::ActiveFilter> FilterBrowser::activeFilters() const {
    std::vector<ave::ActiveFilter> result;
    for (const auto& row : rows_) {
        if (!row.enableBox || !row.enableBox->isChecked()) continue;
        ave::ActiveFilter af;
        af.id      = row.filterId;
        af.enabled = true;
        for (const auto& pr : row.paramRows) {
            if (pr.spin) {
                af.paramValues[pr.key] = pr.spin->value();
            }
        }
        result.push_back(std::move(af));
    }
    return result;
}

void FilterBrowser::setFilterEnabled(const std::string& id, bool enabled) {
    for (auto& row : rows_) {
        if (row.filterId == id && row.enableBox) {
            row.enableBox->setChecked(enabled);
            return;
        }
    }
}

void FilterBrowser::setFilterParam(const std::string& id,
                                   const std::string& key, double value) {
    for (auto& row : rows_) {
        if (row.filterId != id) continue;
        for (auto& pr : row.paramRows) {
            if (pr.key == key && pr.spin) {
                pr.spin->setValue(value);
                return;
            }
        }
    }
}

int FilterBrowser::enabledCount() const {
    int n = 0;
    for (const auto& row : rows_) {
        if (row.enableBox && row.enableBox->isChecked()) { ++n; }
    }
    return n;
}

#endif // ── entire file disabled ──────────────────────────────
