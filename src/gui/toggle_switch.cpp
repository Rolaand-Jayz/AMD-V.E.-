#include "toggle_switch.hpp"

#include <QEnterEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPropertyAnimation>
#include <QHBoxLayout>
#include <QLabel>

ToggleSwitch::ToggleSwitch(QWidget* parent)
    : QAbstractButton(parent) {
    init();
}

ToggleSwitch::ToggleSwitch(const QString& text, QWidget* parent)
    : QAbstractButton(parent) {
    setText(text);
    init();
}

void ToggleSwitch::init() {
    setCheckable(true);
    setCursor(Qt::PointingHandCursor);
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

    anim_ = new QPropertyAnimation(this, "thumbPosition", this);
    anim_->setDuration(160);
    anim_->setEasingCurve(QEasingCurve::InOutQuad);

    thumbPos_ = isChecked() ? 1.0f : 0.0f;

    connect(this, &QAbstractButton::toggled, this, [this](bool checked) {
        updateAnimation();
        (void)checked;
    });
}

QSize ToggleSwitch::sizeHint() const {
    const QFontMetrics fm(font());
    const QString t = text();
    const int textW = t.isEmpty() ? 0 : fm.horizontalAdvance(t) + 6;
    return { kTrackW + textW, kTrackH };
}

QSize ToggleSwitch::minimumSizeHint() const {
    return { kTrackW, kTrackH };
}

float ToggleSwitch::thumbPosition() const { return thumbPos_; }

void ToggleSwitch::setThumbPosition(float pos) {
    thumbPos_ = pos;
    update();
}

void ToggleSwitch::updateAnimation() {
    if (anim_->state() == QAbstractAnimation::Running) { anim_->stop(); }
    anim_->setStartValue(thumbPos_);
    anim_->setEndValue(isChecked() ? 1.0f : 0.0f);
    anim_->start();
}

void ToggleSwitch::paintEvent(QPaintEvent* /*e*/) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // ── Track ──────────────────────────────────────────────────
    const QColor& trackFrom = trackOffColor_;
    const QColor& trackTo   = trackOnColor_;
    QColor track;
    track.setRedF(  trackFrom.redF()   + (trackTo.redF()   - trackFrom.redF())   * thumbPos_);
    track.setGreenF(trackFrom.greenF() + (trackTo.greenF() - trackFrom.greenF()) * thumbPos_);
    track.setBlueF( trackFrom.blueF()  + (trackTo.blueF()  - trackFrom.blueF())  * thumbPos_);
    track.setAlphaF(static_cast<float>(hovered_ ? 0.95 : 0.85));

    const QRectF trackRect(0, (height() - kTrackH) / 2.0, kTrackW, kTrackH);
    QPainterPath trackPath;
    trackPath.addRoundedRect(trackRect, kTrackH / 2.0, kTrackH / 2.0);
    p.fillPath(trackPath, track);

    // ── Thumb ──────────────────────────────────────────────────
    const double thumbTravel = kTrackW - kThumbSize - 2 * kPad;
    const double thumbX      = kPad + thumbTravel * thumbPos_;
    const double thumbY      = (height() - kThumbSize) / 2.0;

    p.setBrush(thumbColor_);
    p.setPen(Qt::NoPen);
    p.drawEllipse(QRectF(thumbX, thumbY, kThumbSize, kThumbSize));

    // ── Optional label text ────────────────────────────────────
    const QString t = text();
    if (!t.isEmpty()) {
        p.setPen(palette().windowText().color());
        const QRect textRect(kTrackW + 6, 0, width() - kTrackW - 6, height());
        p.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, t);
    }
}

void ToggleSwitch::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        toggle();
    }
    QAbstractButton::mouseReleaseEvent(e);
}

void ToggleSwitch::enterEvent(QEnterEvent* e) {
    hovered_ = true;
    update();
    QAbstractButton::enterEvent(e);
}

void ToggleSwitch::leaveEvent(QEvent* e) {
    hovered_ = false;
    update();
    QAbstractButton::leaveEvent(e);
}

void ToggleSwitch::resizeEvent(QResizeEvent* e) {
    QAbstractButton::resizeEvent(e);
    update();
}
