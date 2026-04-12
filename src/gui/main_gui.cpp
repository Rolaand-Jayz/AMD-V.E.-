#include <QApplication>
#include <QColor>
#include <QIcon>
#include <QLinearGradient>
#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <QPixmap>

#include "main_window.hpp"

namespace {

QIcon buildAppIcon() {
    constexpr int size = 256;
    QPixmap pixmap(size, size);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::SmoothPixmapTransform, true);

    const QRectF outerRect(12.0, 12.0, 232.0, 232.0);
    QLinearGradient bg(28.0, 24.0, 224.0, 232.0);
    bg.setColorAt(0.0, QColor("#132238"));
    bg.setColorAt(0.55, QColor("#0f172a"));
    bg.setColorAt(1.0, QColor("#1e293b"));
    painter.setBrush(bg);
    painter.setPen(QPen(QColor("#ea580c"), 8.0));
    painter.drawRoundedRect(outerRect, 46.0, 46.0);

    const QRectF screenRect(40.0, 50.0, 176.0, 112.0);
    painter.setBrush(QColor("#0b1220"));
    painter.setPen(QPen(QColor("#334155"), 4.0));
    painter.drawRoundedRect(screenRect, 24.0, 24.0);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#475569"));
    for (int y : {60, 82, 104, 126}) {
        painter.drawRoundedRect(QRectF(50.0, static_cast<double>(y), 12.0, 14.0), 4.0, 4.0);
        painter.drawRoundedRect(QRectF(194.0, static_cast<double>(y), 12.0, 14.0), 4.0, 4.0);
    }

    QRadialGradient glow(QPointF(170.0, 172.0), 42.0);
    glow.setColorAt(0.0, QColor(253, 186, 116, 225));
    glow.setColorAt(1.0, QColor(245, 158, 11, 32));
    painter.setBrush(glow);
    painter.drawEllipse(QPointF(170.0, 172.0), 42.0, 42.0);

    QPainterPath playPath;
    playPath.moveTo(103.0, 80.0);
    playPath.lineTo(103.0, 132.0);
    playPath.lineTo(154.0, 106.0);
    playPath.closeSubpath();
    QLinearGradient accent(82.0, 72.0, 180.0, 166.0);
    accent.setColorAt(0.0, QColor("#fb923c"));
    accent.setColorAt(0.5, QColor("#f97316"));
    accent.setColorAt(1.0, QColor("#facc15"));
    painter.setBrush(accent);
    painter.drawPath(playPath);

    QPainterPath spark;
    spark.moveTo(176.0, 56.0);
    spark.lineTo(182.0, 72.0);
    spark.lineTo(198.0, 78.0);
    spark.lineTo(182.0, 84.0);
    spark.lineTo(176.0, 100.0);
    spark.lineTo(170.0, 84.0);
    spark.lineTo(154.0, 78.0);
    spark.lineTo(170.0, 72.0);
    spark.closeSubpath();
    painter.setBrush(QColor("#fde68a"));
    painter.drawPath(spark);

    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(QColor("#f97316"), 10.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    QPainterPath outerArc;
    outerArc.moveTo(68.0, 194.0);
    outerArc.cubicTo(88.0, 176.0, 110.0, 166.0, 132.0, 166.0);
    outerArc.cubicTo(154.0, 166.0, 176.0, 176.0, 194.0, 194.0);
    painter.drawPath(outerArc);

    painter.setPen(QPen(QColor("#fdba74"), 8.0, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    QPainterPath innerArc;
    innerArc.moveTo(82.0, 206.0);
    innerArc.cubicTo(98.0, 191.0, 114.0, 184.0, 132.0, 184.0);
    innerArc.cubicTo(150.0, 184.0, 166.0, 191.0, 180.0, 206.0);
    painter.drawPath(innerArc);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#fb923c"));
    QPainterPath cornerA;
    cornerA.moveTo(84.0, 36.0);
    cornerA.lineTo(98.0, 36.0);
    cornerA.lineTo(84.0, 50.0);
    cornerA.closeSubpath();
    painter.drawPath(cornerA);

    painter.setBrush(QColor("#facc15"));
    QPainterPath cornerB;
    cornerB.moveTo(158.0, 206.0);
    cornerB.lineTo(172.0, 206.0);
    cornerB.lineTo(172.0, 220.0);
    cornerB.closeSubpath();
    painter.drawPath(cornerB);

    painter.end();

    QIcon icon;
    icon.addPixmap(pixmap);
    return icon;
}

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationDisplayName(QStringLiteral("AMD Video Enhancer"));
    const QIcon icon = buildAppIcon();
    app.setWindowIcon(icon);
    MainWindow window;
    window.setWindowIcon(icon);
    window.show();
    return app.exec();
}
