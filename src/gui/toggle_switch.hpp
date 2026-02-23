#pragma once

#include <QAbstractButton>
#include <QColor>
#include <QPropertyAnimation>

// ─────────────────────────────────────────────────────────────────
// ToggleSwitch
// ─────────────────────────────────────────────────────────────────
// A pill-shaped toggle button that replaces QCheckBox throughout
// the application.  Animates the thumb position when toggled.
// ─────────────────────────────────────────────────────────────────
class ToggleSwitch final : public QAbstractButton {
    Q_OBJECT
    Q_PROPERTY(float thumbPosition READ thumbPosition WRITE setThumbPosition)

  public:
    explicit ToggleSwitch(QWidget* parent = nullptr);
    explicit ToggleSwitch(const QString& text, QWidget* parent = nullptr);

    QSize sizeHint()    const override;
    QSize minimumSizeHint() const override;

    float thumbPosition() const;
    void  setThumbPosition(float pos);

  protected:
    void paintEvent(QPaintEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void enterEvent(QEnterEvent* e) override;
    void leaveEvent(QEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

  private:
    void init();
    void updateAnimation();

    float              thumbPos_ = 0.0f; // 0.0 = off, 1.0 = on
    bool               hovered_  = false;
    QPropertyAnimation* anim_    = nullptr;

    // Visual constants
    static constexpr int kTrackW    = 44;
    static constexpr int kTrackH    = 24;
    static constexpr int kThumbSize = 18;
    static constexpr int kPad       = 3;

    QColor trackOnColor_  {  82, 130, 255 };
    QColor trackOffColor_ { 120, 120, 120 };
    QColor thumbColor_    { 255, 255, 255 };
};
