#include "MainWindow.h"

#include <QAbstractItemView>
#include <QAction>
#include <QByteArray>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImage>
#include <QIODevice>
#include <QLabel>
#include <QLineEdit>
#include <QList>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMap>
#include <QMenu>
#include <QMessageBox>
#include <QPixmap>
#include <QPoint>
#include <QPushButton>
#include <QResizeEvent>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QSlider>
#include <QShortcut>
#include <QSpinBox>
#include <QKeySequence>
#include <QStringList>
#include <QStyle>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>

namespace {

int oddKernel(int value, int minimum, int maximum)
{
    value = std::clamp(value, minimum, maximum);
    if (value % 2 == 0) {
        ++value;
    }
    if (value > maximum) {
        value -= 2;
    }
    return std::max(minimum, value);
}

cv::Mat ensure8Bit(const cv::Mat &input)
{
    if (input.empty()) {
        return {};
    }

    if (input.depth() == CV_8U) {
        return input;
    }

    cv::Mat output;
    if (input.depth() == CV_16U) {
        input.convertTo(output, CV_8U, 1.0 / 256.0);
    } else {
        cv::normalize(input, output, 0, 255, cv::NORM_MINMAX, CV_8U);
    }
    return output;
}

cv::Mat toBgr(const cv::Mat &input)
{
    const cv::Mat source = ensure8Bit(input);
    if (source.empty()) {
        return {};
    }

    cv::Mat output;
    switch (source.channels()) {
    case 1:
        cv::cvtColor(source, output, cv::COLOR_GRAY2BGR);
        return output;
    case 3:
        return source.clone();
    case 4:
        cv::cvtColor(source, output, cv::COLOR_BGRA2BGR);
        return output;
    default:
        cv::cvtColor(source.reshape(1), output, cv::COLOR_GRAY2BGR);
        return output;
    }
}

cv::Mat toGray(const cv::Mat &input)
{
    const cv::Mat source = ensure8Bit(input);
    if (source.empty()) {
        return {};
    }

    cv::Mat output;
    switch (source.channels()) {
    case 1:
        return source.clone();
    case 3:
        cv::cvtColor(source, output, cv::COLOR_BGR2GRAY);
        return output;
    case 4:
        cv::cvtColor(source, output, cv::COLOR_BGRA2GRAY);
        return output;
    default:
        return source.reshape(1).clone();
    }
}

QImage matToImage(const cv::Mat &input)
{
    const cv::Mat source = ensure8Bit(input);
    if (source.empty()) {
        return {};
    }

    if (source.channels() == 1) {
        QImage image(source.data, source.cols, source.rows, static_cast<int>(source.step), QImage::Format_Grayscale8);
        return image.copy();
    }

    cv::Mat converted;
    if (source.channels() == 3) {
        cv::cvtColor(source, converted, cv::COLOR_BGR2RGB);
        QImage image(converted.data, converted.cols, converted.rows, static_cast<int>(converted.step), QImage::Format_RGB888);
        return image.copy();
    }

    if (source.channels() == 4) {
        cv::cvtColor(source, converted, cv::COLOR_BGRA2RGBA);
        QImage image(converted.data, converted.cols, converted.rows, static_cast<int>(converted.step), QImage::Format_RGBA8888);
        return image.copy();
    }

    return matToImage(toGray(source));
}

std::vector<uchar> byteArrayToVector(const QByteArray &bytes)
{
    return {reinterpret_cast<const uchar *>(bytes.constData()),
            reinterpret_cast<const uchar *>(bytes.constData() + bytes.size())};
}

void drawWaitingLabel(cv::Mat &image, const QString &text)
{
    cv::putText(image,
                text.toStdString(),
                {18, 36},
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                {255, 255, 255},
                3,
                cv::LINE_AA);
    cv::putText(image,
                text.toStdString(),
                {18, 36},
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                {20, 20, 20},
                1,
                cv::LINE_AA);
}

cv::Mat binaryMask(const cv::Mat &gray, int threshold, bool invert = false)
{
    cv::Mat mask;
    cv::threshold(gray, mask, threshold, 255, invert ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY);
    return mask;
}

void cleanMask(cv::Mat &mask, int openKernel, int closeKernel, int dilation)
{
    if (openKernel > 1) {
        const int size = oddKernel(openKernel, 1, 51);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {size, size});
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    }

    if (closeKernel > 1) {
        const int size = oddKernel(closeKernel, 1, 51);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {size, size});
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    }

    if (dilation > 0) {
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
        cv::dilate(mask, mask, kernel, {-1, -1}, dilation);
    }
}

cv::Mat overlayMask(const cv::Mat &bgr, const cv::Mat &mask, const cv::Scalar &color, double alpha = 0.35)
{
    cv::Mat output = bgr.clone();
    cv::Mat tinted(output.size(), output.type(), color);
    tinted.copyTo(output, mask);
    cv::addWeighted(bgr, 1.0 - alpha, output, alpha, 0.0, output);
    return output;
}

double contourCircularity(const std::vector<cv::Point> &contour)
{
    const double perimeter = cv::arcLength(contour, true);
    if (perimeter <= std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    return 4.0 * CV_PI * cv::contourArea(contour) / (perimeter * perimeter);
}

void drawFilteredContours(cv::Mat &output,
                          const cv::Mat &mask,
                          double minArea,
                          double maxArea,
                          double minCircularity,
                          const cv::Scalar &boxColor)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < minArea || (maxArea > 0.0 && area > maxArea)) {
            continue;
        }
        if (minCircularity > 0.0 && contourCircularity(contour) < minCircularity) {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contour);
        const cv::Moments moments = cv::moments(contour);
        cv::rectangle(output, box, boxColor, 2);
        cv::drawContours(output, std::vector<std::vector<cv::Point>>{contour}, -1, {255, 190, 40}, 1);
        if (std::abs(moments.m00) > std::numeric_limits<double>::epsilon()) {
            const cv::Point center(static_cast<int>(std::round(moments.m10 / moments.m00)),
                                   static_cast<int>(std::round(moments.m01 / moments.m00)));
            cv::drawMarker(output, center, {20, 40, 255}, cv::MARKER_CROSS, 14, 2, cv::LINE_AA);
        }
    }
}

} // namespace

class ImageView final : public QLabel {
public:
    explicit ImageView(QWidget *parent = nullptr)
        : QLabel(parent)
    {
        setAlignment(Qt::AlignCenter);
        setMinimumSize(300, 220);
        setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        setText("No media loaded");
        setObjectName("imageView");
    }

    void setImage(const QImage &image)
    {
        image_ = image;
        updatePixmap();
    }

protected:
    void resizeEvent(QResizeEvent *event) override
    {
        QLabel::resizeEvent(event);
        updatePixmap();
    }

private:
    void updatePixmap()
    {
        if (image_.isNull()) {
            clear();
            setText("No media loaded");
            return;
        }

        setText({});
        setPixmap(QPixmap::fromImage(image_).scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    QImage image_;
};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    frameTimer_.setTimerType(Qt::PreciseTimer);
    connect(&frameTimer_, &QTimer::timeout, this, &MainWindow::advanceVideo);

    buildUi();
    buildOperationTree();
    updatePipelineList();
    selectOperation(OperationId::Original);
    updatePipelineControls();
    updatePlaybackControls();
    updateMediaStatus();
}

void MainWindow::buildUi()
{
    setWindowTitle("OpenCV Operation Viewer");
    resize(1280, 820);

    auto *centralWidget = new QWidget(this);
    auto *rootLayout = new QHBoxLayout(centralWidget);
    rootLayout->setContentsMargins(12, 12, 12, 12);
    rootLayout->setSpacing(12);

    auto *leftPanel = new QWidget(centralWidget);
    leftPanel->setObjectName("leftPanel");
    leftPanel->setMinimumWidth(300);
    leftPanel->setMaximumWidth(380);

    auto *leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(14, 14, 14, 14);
    leftLayout->setSpacing(10);

    openButton_ = new QPushButton(style()->standardIcon(QStyle::SP_DialogOpenButton), "Open media", leftPanel);
    openButton_->setMinimumHeight(36);
    connect(openButton_, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
            this,
            "Open image or video",
            QString(),
            "Media files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.mp4 *.mov *.avi *.mkv *.webm *.m4v);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;"
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;"
            "All files (*)");

        if (!path.isEmpty()) {
            loadMedia(path);
        }
    });

    mediaStatusLabel_ = new QLabel("No media loaded", leftPanel);
    mediaStatusLabel_->setObjectName("metaLabel");
    mediaStatusLabel_->setWordWrap(true);

    operationFilter_ = new QLineEdit(leftPanel);
    operationFilter_->setPlaceholderText("Filter operations");
    operationFilter_->setClearButtonEnabled(true);
    connect(operationFilter_, &QLineEdit::textChanged, this, &MainWindow::filterOperations);

    operationTree_ = new QTreeWidget(leftPanel);
    operationTree_->setHeaderHidden(true);
    operationTree_->setRootIsDecorated(true);
    operationTree_->setUniformRowHeights(true);
    operationTree_->setMinimumHeight(280);
    operationTree_->header()->setStretchLastSection(true);
    connect(operationTree_, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem *item) {
        if (!item || item->childCount() > 0) {
            return;
        }

        selectOperation(static_cast<OperationId>(item->data(0, Qt::UserRole).toInt()));
    });
    connect(operationTree_, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem *item) {
        if (!item || item->childCount() > 0) {
            return;
        }

        selectOperation(static_cast<OperationId>(item->data(0, Qt::UserRole).toInt()));
        addSelectedOperationToChain();
    });

    operationDescriptionLabel_ = new QLabel(leftPanel);
    operationDescriptionLabel_->setObjectName("descriptionLabel");
    operationDescriptionLabel_->setWordWrap(true);

    auto *chainTitleLabel = new QLabel("Operation chain", leftPanel);
    chainTitleLabel->setObjectName("viewerTitle");

    pipelineList_ = new QListWidget(leftPanel);
    pipelineList_->setMinimumHeight(130);
    pipelineList_->setAlternatingRowColors(true);
    pipelineList_->setSelectionMode(QAbstractItemView::SingleSelection);
    pipelineList_->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(pipelineList_, &QListWidget::currentRowChanged, this, &MainWindow::selectPipelineStep);
    connect(pipelineList_, &QListWidget::customContextMenuRequested, this, [this](const QPoint &position) {
        const QModelIndex index = pipelineList_->indexAt(position);
        if (!index.isValid()) {
            return;
        }

        pipelineList_->setCurrentRow(index.row());

        QMenu menu(this);
        QAction *removeAction = menu.addAction(style()->standardIcon(QStyle::SP_DialogCancelButton), "Remove selected operation");
        connect(removeAction, &QAction::triggered, this, &MainWindow::removeSelectedPipelineStep);
        menu.exec(pipelineList_->viewport()->mapToGlobal(position));
    });

    auto *deleteShortcut = new QShortcut(QKeySequence::Delete, pipelineList_);
    connect(deleteShortcut, &QShortcut::activated, this, &MainWindow::removeSelectedPipelineStep);

    auto *backspaceShortcut = new QShortcut(QKeySequence::Backspace, pipelineList_);
    connect(backspaceShortcut, &QShortcut::activated, this, &MainWindow::removeSelectedPipelineStep);

    auto *chainButtonRow = new QWidget(leftPanel);
    auto *chainButtonLayout = new QHBoxLayout(chainButtonRow);
    chainButtonLayout->setContentsMargins(0, 0, 0, 0);
    chainButtonLayout->setSpacing(6);

    addStepButton_ = new QPushButton(style()->standardIcon(QStyle::SP_DialogApplyButton), "Add", chainButtonRow);
    removeStepButton_ = new QPushButton(style()->standardIcon(QStyle::SP_DialogCancelButton), "Remove", chainButtonRow);
    moveStepUpButton_ = new QPushButton(style()->standardIcon(QStyle::SP_ArrowUp), "", chainButtonRow);
    moveStepDownButton_ = new QPushButton(style()->standardIcon(QStyle::SP_ArrowDown), "", chainButtonRow);
    clearChainButton_ = new QPushButton(style()->standardIcon(QStyle::SP_DialogResetButton), "", chainButtonRow);

    addStepButton_->setToolTip("Add selected operation to chain");
    removeStepButton_->setToolTip("Remove selected chain step");
    moveStepUpButton_->setToolTip("Move selected step up");
    moveStepDownButton_->setToolTip("Move selected step down");
    clearChainButton_->setToolTip("Clear operation chain");

    moveStepUpButton_->setFixedWidth(34);
    moveStepDownButton_->setFixedWidth(34);
    clearChainButton_->setFixedWidth(34);

    connect(addStepButton_, &QPushButton::clicked, this, &MainWindow::addSelectedOperationToChain);
    connect(removeStepButton_, &QPushButton::clicked, this, &MainWindow::removeSelectedPipelineStep);
    connect(moveStepUpButton_, &QPushButton::clicked, this, [this]() { moveSelectedPipelineStep(-1); });
    connect(moveStepDownButton_, &QPushButton::clicked, this, [this]() { moveSelectedPipelineStep(1); });
    connect(clearChainButton_, &QPushButton::clicked, this, &MainWindow::clearPipeline);

    chainButtonLayout->addWidget(addStepButton_, 1);
    chainButtonLayout->addWidget(removeStepButton_);
    chainButtonLayout->addWidget(moveStepUpButton_);
    chainButtonLayout->addWidget(moveStepDownButton_);
    chainButtonLayout->addWidget(clearChainButton_);

    parameterGroup_ = new QGroupBox("Parameters", leftPanel);
    auto *parameterLayout = new QVBoxLayout(parameterGroup_);
    parameterLayout->setContentsMargins(10, 14, 10, 10);
    parameterLayout->setSpacing(8);

    for (int index = 0; index < ParameterCount; ++index) {
        auto &widgets = parameterWidgets_[index];
        widgets.row = new QWidget(parameterGroup_);
        auto *rowLayout = new QHBoxLayout(widgets.row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(8);

        widgets.label = new QLabel(parameterGroup_);
        widgets.label->setMinimumWidth(92);
        widgets.slider = new QSlider(Qt::Horizontal, parameterGroup_);
        widgets.spinBox = new QSpinBox(parameterGroup_);
        widgets.spinBox->setMinimumWidth(72);

        rowLayout->addWidget(widgets.label);
        rowLayout->addWidget(widgets.slider, 1);
        rowLayout->addWidget(widgets.spinBox);
        parameterLayout->addWidget(widgets.row);

        connect(widgets.slider, &QSlider::valueChanged, this, [this, index](int value) {
            if (configuringParameters_) {
                return;
            }
            {
                QSignalBlocker blocker(parameterWidgets_[index].spinBox);
                parameterWidgets_[index].spinBox->setValue(value);
            }
            storeParameterValue(index, value);
            processCurrentFrame();
        });

        connect(widgets.spinBox, qOverload<int>(&QSpinBox::valueChanged), this, [this, index](int value) {
            if (configuringParameters_) {
                return;
            }
            {
                QSignalBlocker blocker(parameterWidgets_[index].slider);
                parameterWidgets_[index].slider->setValue(value);
            }
            storeParameterValue(index, value);
            processCurrentFrame();
        });
    }

    leftLayout->addWidget(openButton_);
    leftLayout->addWidget(mediaStatusLabel_);
    leftLayout->addSpacing(8);
    leftLayout->addWidget(operationFilter_);
    leftLayout->addWidget(operationTree_, 1);
    leftLayout->addWidget(chainTitleLabel);
    leftLayout->addWidget(pipelineList_);
    leftLayout->addWidget(chainButtonRow);
    leftLayout->addWidget(operationDescriptionLabel_);
    leftLayout->addWidget(parameterGroup_);

    auto *mainPanel = new QWidget(centralWidget);
    auto *mainLayout = new QVBoxLayout(mainPanel);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(10);

    auto *stageHeaderLabel = new QLabel("Pipeline stages", mainPanel);
    stageHeaderLabel->setObjectName("viewerTitle");

    stageScrollArea_ = new QScrollArea(mainPanel);
    stageScrollArea_->setObjectName("stageScrollArea");
    stageScrollArea_->setWidgetResizable(true);

    stageContainer_ = new QWidget(stageScrollArea_);
    stageContainer_->setObjectName("stageContainer");
    stageGridLayout_ = new QGridLayout(stageContainer_);
    stageGridLayout_->setContentsMargins(0, 0, 0, 0);
    stageGridLayout_->setHorizontalSpacing(10);
    stageGridLayout_->setVerticalSpacing(10);
    stageGridLayout_->setColumnStretch(0, 1);
    stageGridLayout_->setColumnStretch(1, 1);

    stageScrollArea_->setWidget(stageContainer_);

    auto *transport = new QWidget(mainPanel);
    transport->setObjectName("transport");
    auto *transportLayout = new QHBoxLayout(transport);
    transportLayout->setContentsMargins(12, 10, 12, 10);
    transportLayout->setSpacing(8);

    previousFrameButton_ = new QPushButton(style()->standardIcon(QStyle::SP_MediaSeekBackward), "", transport);
    playPauseButton_ = new QPushButton(style()->standardIcon(QStyle::SP_MediaPlay), "", transport);
    stopButton_ = new QPushButton(style()->standardIcon(QStyle::SP_MediaStop), "", transport);
    nextFrameButton_ = new QPushButton(style()->standardIcon(QStyle::SP_MediaSeekForward), "", transport);

    previousFrameButton_->setToolTip("Previous frame");
    playPauseButton_->setToolTip("Play or pause");
    stopButton_->setToolTip("Stop and rewind");
    nextFrameButton_->setToolTip("Next frame");

    const QList<QPushButton *> transportButtons = {previousFrameButton_, playPauseButton_, stopButton_, nextFrameButton_};
    for (auto *button : transportButtons) {
        button->setFixedSize(36, 32);
    }

    timelineSlider_ = new QSlider(Qt::Horizontal, transport);
    timelineSlider_->setEnabled(false);

    frameLabel_ = new QLabel("Frame -", transport);
    frameLabel_->setObjectName("metaLabel");
    frameLabel_->setMinimumWidth(170);
    frameLabel_->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    connect(previousFrameButton_, &QPushButton::clicked, this, [this]() { stepVideo(-1); });
    connect(playPauseButton_, &QPushButton::clicked, this, [this]() { setPlaying(!isPlaying_); });
    connect(stopButton_, &QPushButton::clicked, this, [this]() {
        setPlaying(false);
        seekVideo(0);
    });
    connect(nextFrameButton_, &QPushButton::clicked, this, [this]() { stepVideo(1); });
    connect(timelineSlider_, &QSlider::sliderPressed, this, [this]() { setPlaying(false); });
    connect(timelineSlider_, &QSlider::sliderReleased, this, [this]() { seekVideo(timelineSlider_->value()); });
    connect(timelineSlider_, &QSlider::sliderMoved, this, [this](int value) {
        if (frameCount_ > 0) {
            frameLabel_->setText(QString("Frame %1 / %2").arg(value + 1).arg(frameCount_));
        }
    });

    transportLayout->addWidget(previousFrameButton_);
    transportLayout->addWidget(playPauseButton_);
    transportLayout->addWidget(stopButton_);
    transportLayout->addWidget(nextFrameButton_);
    transportLayout->addWidget(timelineSlider_, 1);
    transportLayout->addWidget(frameLabel_);

    mainLayout->addWidget(stageHeaderLabel);
    mainLayout->addWidget(stageScrollArea_, 1);
    mainLayout->addWidget(transport);

    rootLayout->addWidget(leftPanel);
    rootLayout->addWidget(mainPanel, 1);

    setCentralWidget(centralWidget);

    setStyleSheet(R"(
        QMainWindow {
            background: #f4f5f7;
            color: #1f2933;
        }
        QWidget#leftPanel,
        QWidget#transport {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 8px;
        }
        QScrollArea#stageScrollArea {
            background: transparent;
            border: 0;
        }
        QWidget#stageContainer {
            background: transparent;
        }
        QWidget#stagePane {
            background: #ffffff;
            border: 1px solid #d9dee7;
            border-radius: 8px;
        }
        QLabel#viewerTitle {
            color: #253041;
            font-size: 14px;
            font-weight: 700;
        }
        QLabel#metaLabel {
            color: #637083;
            font-size: 12px;
        }
        QLabel#descriptionLabel {
            color: #364152;
            background: #eef3f8;
            border: 1px solid #d5e0eb;
            border-radius: 6px;
            padding: 8px;
        }
        QLabel#imageView {
            background: #111827;
            border: 1px solid #2f3b4f;
            border-radius: 6px;
            color: #9aa7b8;
        }
        QLineEdit,
        QSpinBox {
            border: 1px solid #cfd7e3;
            border-radius: 6px;
            padding: 5px 7px;
            background: #ffffff;
        }
        QPushButton {
            border: 1px solid #c8d1de;
            border-radius: 6px;
            padding: 6px 10px;
            background: #f8fafc;
        }
        QPushButton:hover {
            background: #eef4fb;
        }
        QPushButton:pressed {
            background: #dfe9f5;
        }
        QPushButton:disabled {
            color: #98a4b3;
            background: #f2f4f7;
        }
        QTreeWidget,
        QListWidget {
            border: 1px solid #d6dde8;
            border-radius: 6px;
            background: #ffffff;
            outline: 0;
        }
        QTreeWidget::item,
        QListWidget::item {
            min-height: 26px;
            padding: 2px 4px;
        }
        QTreeWidget::item:selected,
        QListWidget::item:selected {
            background: #2f6f9f;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #d6dde8;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: 700;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
        }
    )");
}

void MainWindow::buildOperationTree()
{
    operationTree_->clear();

    QMap<QString, QTreeWidgetItem *> categories;
    for (const OperationInfo &operation : operations()) {
        auto *category = categories.value(operation.category, nullptr);
        if (!category) {
            category = new QTreeWidgetItem(operationTree_, QStringList{operation.category});
            category->setFirstColumnSpanned(true);
            categories.insert(operation.category, category);
        }

        auto *item = new QTreeWidgetItem(category, QStringList{operation.name});
        item->setData(0, Qt::UserRole, static_cast<int>(operation.id));
        item->setToolTip(0, operation.description);
    }

    operationTree_->expandAll();

    const QList<QTreeWidgetItem *> matches = operationTree_->findItems("Original", Qt::MatchRecursive);
    if (!matches.isEmpty()) {
        operationTree_->setCurrentItem(matches.first());
    }
}

void MainWindow::configureParameters(OperationId id, const QVector<int> &values)
{
    configuringParameters_ = true;

    for (auto &widgets : parameterWidgets_) {
        widgets.row->hide();
    }

    const QVector<ParameterSpec> specs = parameterSpecs(id);
    for (int index = 0; index < specs.size() && index < ParameterCount; ++index) {
        const ParameterSpec &spec = specs[index];
        auto &widgets = parameterWidgets_[index];

        widgets.label->setText(spec.label);
        widgets.slider->setRange(spec.minimum, spec.maximum);
        widgets.spinBox->setRange(spec.minimum, spec.maximum);
        const int value = index < values.size()
            ? std::clamp(values[index], spec.minimum, spec.maximum)
            : spec.defaultValue;
        widgets.slider->setValue(value);
        widgets.spinBox->setValue(value);
        parameterValues_[index] = value;
        widgets.row->show();
    }

    parameterGroup_->setVisible(!specs.isEmpty());
    configuringParameters_ = false;
}

void MainWindow::filterOperations(const QString &filter)
{
    const QString needle = filter.trimmed();

    for (int topIndex = 0; topIndex < operationTree_->topLevelItemCount(); ++topIndex) {
        QTreeWidgetItem *category = operationTree_->topLevelItem(topIndex);
        bool categoryHasMatch = needle.isEmpty() || category->text(0).contains(needle, Qt::CaseInsensitive);

        for (int childIndex = 0; childIndex < category->childCount(); ++childIndex) {
            QTreeWidgetItem *child = category->child(childIndex);
            const auto id = static_cast<OperationId>(child->data(0, Qt::UserRole).toInt());
            const OperationInfo *info = operationInfo(id);
            const bool childMatches = needle.isEmpty()
                || child->text(0).contains(needle, Qt::CaseInsensitive)
                || (info && info->description.contains(needle, Qt::CaseInsensitive));

            child->setHidden(!childMatches && !categoryHasMatch);
            categoryHasMatch = categoryHasMatch || childMatches;
        }

        category->setHidden(!categoryHasMatch);
        category->setExpanded(!needle.isEmpty() || category->isExpanded());
    }
}

void MainWindow::loadMedia(const QString &path)
{
    resetMedia();
    currentPath_ = path;

    QFile file(path);
    if (file.open(QIODevice::ReadOnly)) {
        const QByteArray bytes = file.readAll();
        const std::vector<uchar> encoded = byteArrayToVector(bytes);
        currentFrame_ = cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
    }

    if (!currentFrame_.empty()) {
        isVideo_ = false;
        frameIndex_ = 0;
        frameCount_ = 1;
        framesPerSecond_ = 0.0;
        timelineSlider_->setRange(0, 0);
        processCurrentFrame();
        updatePlaybackControls();
        updateMediaStatus();
        return;
    }

    capture_.open(path.toStdString());
    if (!capture_.isOpened()) {
        resetMedia();
        QMessageBox::warning(this, "Open media", "Could not open the selected file as an image or video.");
        return;
    }

    isVideo_ = true;
    framesPerSecond_ = capture_.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(framesPerSecond_) || framesPerSecond_ <= 0.0) {
        framesPerSecond_ = 30.0;
    }

    const double rawFrameCount = capture_.get(cv::CAP_PROP_FRAME_COUNT);
    frameCount_ = std::isfinite(rawFrameCount) && rawFrameCount > 0.0
        ? static_cast<int>(rawFrameCount)
        : 0;
    timelineSlider_->setRange(0, std::max(0, frameCount_ - 1));

    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        resetMedia();
        QMessageBox::warning(this, "Open media", "The selected video opened but no frames could be decoded.");
        return;
    }

    currentFrame_ = frame;
    frameIndex_ = 0;
    processCurrentFrame();
    updatePlaybackControls();
    updateMediaStatus();
}

void MainWindow::resetMedia()
{
    setPlaying(false);
    capture_.release();
    currentFrame_.release();
    currentPath_.clear();
    isVideo_ = false;
    frameIndex_ = 0;
    frameCount_ = 0;
    framesPerSecond_ = 30.0;
    timelineSlider_->setRange(0, 0);
    timelineSlider_->setValue(0);
    resetTemporalState();
}

void MainWindow::resetTemporalState()
{
    resetPreviewState();
    for (PipelineStep &step : pipeline_) {
        resetStepState(step);
    }
}

void MainWindow::resetTemporalStateFrom(int firstStep)
{
    if (firstStep < 0) {
        resetTemporalState();
        return;
    }

    for (int index = firstStep; index < pipeline_.size(); ++index) {
        resetStepState(pipeline_[index]);
    }
}

void MainWindow::resetPreviewState()
{
    resetStepState(previewStep_);
}

void MainWindow::resetStepState(PipelineStep &step)
{
    step.previousGrayFrame.release();
    step.runningAverage.release();
    step.previousPoints.clear();
    step.backgroundSubtractor.release();
    step.backgroundHistory = 0;
    step.backgroundThreshold = 0;
}

void MainWindow::selectOperation(OperationId id)
{
    selectedOperation_ = id;
    selectedStepIndex_ = -1;

    if (pipelineList_) {
        QSignalBlocker blocker(pipelineList_);
        pipelineList_->setCurrentRow(-1);
    }

    configureParameters(id, defaultParameters(id));
    previewStep_.id = id;
    previewStep_.parameters = parameterValues_;
    resetPreviewState();

    if (const OperationInfo *info = operationInfo(id)) {
        operationDescriptionLabel_->setText(info->description);
    }

    processCurrentFrame();
    updatePipelineControls();
}

void MainWindow::selectPipelineStep(int row)
{
    if (updatingPipelineList_) {
        return;
    }

    if (row < 0 || row >= pipeline_.size()) {
        selectedStepIndex_ = -1;
        configureParameters(selectedOperation_, defaultParameters(selectedOperation_));
        previewStep_.id = selectedOperation_;
        previewStep_.parameters = parameterValues_;
        if (const OperationInfo *info = operationInfo(selectedOperation_)) {
            operationDescriptionLabel_->setText(info->description);
        }
        updatePipelineControls();
        processCurrentFrame();
        return;
    }

    selectedStepIndex_ = row;
    PipelineStep &step = pipeline_[row];
    selectedOperation_ = step.id;
    configureParameters(step.id, step.parameters);

    if (const OperationInfo *info = operationInfo(step.id)) {
        const QString summary = parameterSummary(step.id, step.parameters);
        operationDescriptionLabel_->setText(summary.isEmpty()
                                                ? QString("Step %1: %2\n%3").arg(row + 1).arg(info->name, info->description)
                                                : QString("Step %1: %2\n%3\n%4").arg(row + 1).arg(info->name, info->description, summary));
    }

    updatePipelineControls();
    processCurrentFrame();
}

void MainWindow::addSelectedOperationToChain()
{
    PipelineStep step;
    step.id = selectedOperation_;
    step.parameters = parameterValues_;
    pipeline_.append(step);
    selectedStepIndex_ = pipeline_.size() - 1;

    updatePipelineList();
    resetTemporalStateFrom(selectedStepIndex_);
    selectPipelineStep(selectedStepIndex_);
    processCurrentFrame();
}

void MainWindow::removeSelectedPipelineStep()
{
    if (!hasSelectedPipelineStep()) {
        return;
    }

    const int removedIndex = selectedStepIndex_;
    pipeline_.removeAt(removedIndex);
    selectedStepIndex_ = pipeline_.isEmpty()
        ? -1
        : std::min(removedIndex, static_cast<int>(pipeline_.size()) - 1);

    updatePipelineList();
    resetTemporalStateFrom(removedIndex);

    if (hasSelectedPipelineStep()) {
        selectPipelineStep(selectedStepIndex_);
    } else {
        selectPipelineStep(-1);
    }
    processCurrentFrame();
}

void MainWindow::moveSelectedPipelineStep(int direction)
{
    if (!hasSelectedPipelineStep() || direction == 0) {
        return;
    }

    const int targetIndex = selectedStepIndex_ + direction;
    if (targetIndex < 0 || targetIndex >= pipeline_.size()) {
        return;
    }

    pipeline_.move(selectedStepIndex_, targetIndex);
    selectedStepIndex_ = targetIndex;

    updatePipelineList();
    resetTemporalStateFrom(std::min(selectedStepIndex_, selectedStepIndex_ - direction));
    selectPipelineStep(selectedStepIndex_);
    processCurrentFrame();
}

void MainWindow::clearPipeline()
{
    if (pipeline_.isEmpty()) {
        return;
    }

    pipeline_.clear();
    selectedStepIndex_ = -1;
    updatePipelineList();
    resetTemporalState();
    selectPipelineStep(-1);
    processCurrentFrame();
}

void MainWindow::updatePipelineList()
{
    if (!pipelineList_) {
        return;
    }

    updatingPipelineList_ = true;
    QSignalBlocker blocker(pipelineList_);
    pipelineList_->clear();

    for (int index = 0; index < pipeline_.size(); ++index) {
        const PipelineStep &step = pipeline_[index];
        const OperationInfo *info = operationInfo(step.id);
        auto *item = new QListWidgetItem(pipelineStepTitle(index, step), pipelineList_);
        if (info) {
            const QString summary = parameterSummary(step.id, step.parameters);
            item->setToolTip(summary.isEmpty()
                                 ? info->description
                                 : QString("%1\n%2").arg(info->description, summary));
        }
    }

    if (hasSelectedPipelineStep()) {
        pipelineList_->setCurrentRow(selectedStepIndex_);
    }
    updatingPipelineList_ = false;
    updatePipelineControls();
}

void MainWindow::updatePipelineControls()
{
    const bool hasSelection = hasSelectedPipelineStep();
    addStepButton_->setEnabled(true);
    removeStepButton_->setEnabled(hasSelection);
    moveStepUpButton_->setEnabled(hasSelection && selectedStepIndex_ > 0);
    moveStepDownButton_->setEnabled(hasSelection && selectedStepIndex_ < pipeline_.size() - 1);
    clearChainButton_->setEnabled(!pipeline_.isEmpty());
}

void MainWindow::storeParameterValue(int index, int value)
{
    if (index < 0 || index >= parameterValues_.size()) {
        return;
    }

    parameterValues_[index] = value;

    if (hasSelectedPipelineStep()) {
        PipelineStep &step = pipeline_[selectedStepIndex_];
        if (step.parameters.isEmpty()) {
            step.parameters = defaultParameters(step.id);
        }
        if (index >= step.parameters.size()) {
            step.parameters.resize(index + 1);
        }
        step.parameters[index] = value;

        resetTemporalStateFrom(selectedStepIndex_);
        updatePipelineList();

        if (const OperationInfo *info = operationInfo(step.id)) {
            const QString summary = parameterSummary(step.id, step.parameters);
            operationDescriptionLabel_->setText(summary.isEmpty()
                                                    ? QString("Step %1: %2\n%3")
                                                          .arg(selectedStepIndex_ + 1)
                                                          .arg(info->name, info->description)
                                                    : QString("Step %1: %2\n%3\n%4")
                                                          .arg(selectedStepIndex_ + 1)
                                                          .arg(info->name, info->description, summary));
        }
        return;
    }

    previewStep_.parameters = parameterValues_;
    resetPreviewState();
}

void MainWindow::processCurrentFrame()
{
    if (currentFrame_.empty()) {
        updateStageImages({"Input"}, {{}});
        updateMediaStatus();
        return;
    }

    QVector<QString> titles;
    QVector<cv::Mat> frames;
    titles.append("Input");
    frames.append(currentFrame_);

    if (pipeline_.isEmpty()) {
        previewStep_.id = selectedOperation_;
        previewStep_.parameters = parameterValues_;
        const OperationInfo *info = operationInfo(selectedOperation_);
        const QString name = info ? info->name : QString("Preview");
        titles.append(QString("Preview - %1").arg(name));
        frames.append(applyOperation(currentFrame_, previewStep_));
    } else {
        cv::Mat stage = currentFrame_;
        for (int index = 0; index < pipeline_.size(); ++index) {
            PipelineStep &step = pipeline_[index];
            const OperationInfo *info = operationInfo(step.id);
            const QString name = info ? info->name : QString("Unknown operation");
            stage = applyOperation(stage, step);
            titles.append(QString("%1. %2").arg(index + 1).arg(name));
            frames.append(stage);
        }
    }

    updateStageImages(titles, frames);
    updateMediaStatus();
}

void MainWindow::updateStageImages(const QVector<QString> &titles, const QVector<cv::Mat> &frames)
{
    rebuildStageViews(titles);

    for (int index = 0; index < stageViews_.size(); ++index) {
        const cv::Mat frame = index < frames.size() ? frames[index] : cv::Mat();
        stageViews_[index]->setImage(matToImage(frame));
    }
}

void MainWindow::rebuildStageViews(const QVector<QString> &titles)
{
    if (titles == stageTitles_) {
        return;
    }

    while (QLayoutItem *item = stageGridLayout_->takeAt(0)) {
        if (QWidget *widget = item->widget()) {
            delete widget;
        }
        delete item;
    }

    stageViews_.clear();
    stageTitles_ = titles;

    for (int index = 0; index < titles.size(); ++index) {
        auto *pane = new QWidget(stageContainer_);
        pane->setObjectName("stagePane");
        auto *layout = new QVBoxLayout(pane);
        layout->setContentsMargins(10, 10, 10, 10);
        layout->setSpacing(7);

        auto *title = new QLabel(titles[index], pane);
        title->setObjectName("viewerTitle");

        auto *view = new ImageView(pane);
        layout->addWidget(title);
        layout->addWidget(view, 1);

        stageGridLayout_->addWidget(pane, index / 2, index % 2);
        stageViews_.append(view);
    }

    stageGridLayout_->setColumnStretch(0, 1);
    stageGridLayout_->setColumnStretch(1, 1);
}

void MainWindow::updateMediaStatus()
{
    if (currentFrame_.empty()) {
        mediaStatusLabel_->setText("No media loaded");
        frameLabel_->setText("Frame -");
        setWindowTitle("OpenCV Operation Viewer");
        return;
    }

    const QFileInfo fileInfo(currentPath_);
    const QString dimensions = QString("%1 x %2").arg(currentFrame_.cols).arg(currentFrame_.rows);

    if (isVideo_) {
        const QString frameText = frameCount_ > 0
            ? QString("Frame %1 / %2").arg(frameIndex_ + 1).arg(frameCount_)
            : QString("Frame %1").arg(frameIndex_ + 1);
        const QString fpsText = QString::number(framesPerSecond_, 'f', 1);

        mediaStatusLabel_->setText(QString("Video: %1\n%2, %3 fps").arg(fileInfo.fileName(), dimensions, fpsText));
        frameLabel_->setText(frameText);
        setWindowTitle(QString("%1 - OpenCV Operation Viewer").arg(fileInfo.fileName()));
    } else {
        mediaStatusLabel_->setText(QString("Image: %1\n%2").arg(fileInfo.fileName(), dimensions));
        frameLabel_->setText("Still image");
        setWindowTitle(QString("%1 - OpenCV Operation Viewer").arg(fileInfo.fileName()));
    }

    QSignalBlocker blocker(timelineSlider_);
    timelineSlider_->setValue(frameIndex_);
}

void MainWindow::updatePlaybackControls()
{
    const bool canScrub = isVideo_ && !currentFrame_.empty();
    previousFrameButton_->setEnabled(canScrub);
    playPauseButton_->setEnabled(canScrub);
    stopButton_->setEnabled(canScrub);
    nextFrameButton_->setEnabled(canScrub);
    timelineSlider_->setEnabled(canScrub && frameCount_ > 1);

    playPauseButton_->setIcon(style()->standardIcon(isPlaying_ ? QStyle::SP_MediaPause : QStyle::SP_MediaPlay));
}

void MainWindow::seekVideo(int frameNumber)
{
    if (!isVideo_ || !capture_.isOpened()) {
        return;
    }

    frameNumber = std::max(0, frameNumber);
    if (frameCount_ > 0) {
        frameNumber = std::min(frameNumber, frameCount_ - 1);
    }

    setPlaying(false);
    resetTemporalState();
    capture_.set(cv::CAP_PROP_POS_FRAMES, frameNumber);

    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        return;
    }

    currentFrame_ = frame;
    frameIndex_ = frameNumber;
    processCurrentFrame();
    updatePlaybackControls();
}

void MainWindow::advanceVideo()
{
    if (!isVideo_ || !capture_.isOpened()) {
        setPlaying(false);
        return;
    }

    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
        setPlaying(false);
        return;
    }

    currentFrame_ = frame;
    const double nextFrame = capture_.get(cv::CAP_PROP_POS_FRAMES);
    if (std::isfinite(nextFrame) && nextFrame > 0.0) {
        frameIndex_ = static_cast<int>(nextFrame) - 1;
    } else {
        ++frameIndex_;
    }

    processCurrentFrame();
    updatePlaybackControls();
}

void MainWindow::setPlaying(bool playing)
{
    if (playing && (!isVideo_ || currentFrame_.empty())) {
        playing = false;
    }

    isPlaying_ = playing;
    if (isPlaying_) {
        const int intervalMs = std::max(1, static_cast<int>(std::round(1000.0 / framesPerSecond_)));
        frameTimer_.start(intervalMs);
    } else {
        frameTimer_.stop();
    }

    updatePlaybackControls();
}

void MainWindow::stepVideo(int frameDelta)
{
    if (!isVideo_) {
        return;
    }

    seekVideo(frameIndex_ + frameDelta);
}

QVector<OperationInfo> MainWindow::operations()
{
    return {
        {OperationId::Original, "Fundamentals", "Original", "Pass the input through unchanged."},
        {OperationId::Grayscale, "Fundamentals", "Grayscale", "Convert the input to a single luminance channel."},
        {OperationId::GaussianBlur, "Fundamentals", "Gaussian blur", "Smooth noise with a Gaussian kernel before segmentation or edge detection."},
        {OperationId::MedianBlur, "Fundamentals", "Median blur", "Reduce salt-and-pepper noise while preserving stronger edges."},
        {OperationId::EqualizeHistogram, "Contrast and illumination", "Equalize histogram", "Spread grayscale contrast across the available intensity range."},
        {OperationId::Clahe, "Contrast and illumination", "CLAHE", "Boost local contrast while limiting noise amplification."},
        {OperationId::NormalizeIntensity, "Contrast and illumination", "Normalize intensity", "Normalize luminance to the full 8-bit range."},
        {OperationId::BrightnessContrast, "Contrast and illumination", "Brightness and contrast", "Apply a linear contrast and brightness adjustment."},
        {OperationId::GammaCorrection, "Contrast and illumination", "Gamma correction", "Nonlinear intensity remapping for lifting or suppressing bright regions."},
        {OperationId::IlluminationCorrection, "Contrast and illumination", "Divide by local background", "Suppress slow illumination changes by dividing by a blurred background estimate."},
        {OperationId::DifferenceOfGaussian, "Contrast and illumination", "Difference of Gaussian", "Highlight structures that survive a small blur but not a larger background blur."},
        {OperationId::HsvMask, "Color and masks", "HSV color mask", "Isolate pixels inside a hue, saturation, and value range."},
        {OperationId::BrightMask, "Bright object isolation", "Bright mask", "Keep pixels above a brightness threshold, optionally suppressing highly saturated clutter."},
        {OperationId::AdaptiveBrightMask, "Bright object isolation", "Adaptive bright mask", "Extract pixels brighter than their local background."},
        {OperationId::BrightComponents, "Bright object isolation", "Bright components", "Threshold bright regions and draw bounding boxes around connected candidates."},
        {OperationId::CircularBrightBlobs, "Bright object isolation", "Circular bright blobs", "Find bright components that satisfy area and circularity constraints."},
        {OperationId::BestBrightCandidate, "Bright object isolation", "Best bright candidate", "Score bright components by intensity, area, and circularity, then draw the strongest candidate."},
        {OperationId::SimpleBlobDetector, "Bright object isolation", "Simple blob detector", "Use OpenCV's blob detector to find bright blob-like regions."},
        {OperationId::MserRegions, "Bright object isolation", "MSER bright regions", "Find stable extremal regions and keep bright candidates."},
        {OperationId::HoughCircles, "Bright object isolation", "Hough circles", "Detect circular bright-object candidates after grayscale smoothing."},
        {OperationId::RunningAverageForeground, "Bright object isolation", "Running-average foreground", "Detect moving foreground against a running average background model.", true},
        {OperationId::MotionBrightMask, "Bright object isolation", "Motion-gated bright mask", "Combine frame difference with a brightness threshold to reject static bright clutter.", true},
        {OperationId::Canny, "Edges and thresholds", "Canny edges", "Extract edges using low and high hysteresis thresholds."},
        {OperationId::SobelMagnitude, "Edges and thresholds", "Sobel magnitude", "Show gradient strength from horizontal and vertical Sobel derivatives."},
        {OperationId::LaplacianEdges, "Edges and thresholds", "Laplacian edges", "Highlight second-derivative edge responses."},
        {OperationId::BinaryThreshold, "Edges and thresholds", "Binary threshold", "Convert luminance to a binary foreground/background mask."},
        {OperationId::AdaptiveThreshold, "Edges and thresholds", "Adaptive threshold", "Build a binary mask with local neighborhood thresholds for uneven lighting."},
        {OperationId::Erode, "Morphology", "Erode mask", "Shrink foreground regions in a thresholded mask."},
        {OperationId::Dilate, "Morphology", "Dilate mask", "Expand foreground regions in a thresholded mask."},
        {OperationId::MorphOpen, "Morphology", "Open mask", "Remove small blobs using erosion followed by dilation."},
        {OperationId::MorphClose, "Morphology", "Close mask", "Fill small gaps using dilation followed by erosion."},
        {OperationId::WhiteTopHat, "Morphology", "White top-hat", "Highlight small bright structures against uneven bright backgrounds."},
        {OperationId::BlackHat, "Morphology", "Black-hat", "Highlight small dark gaps or shadow structures after morphological closing."},
        {OperationId::Contours, "Tracking helpers", "Contours and boxes", "Find edge contours and draw bounding boxes over likely objects."},
        {OperationId::ConnectedComponents, "Tracking helpers", "Connected components", "Draw boxes and centers for connected regions in a binary mask."},
        {OperationId::FastCorners, "Tracking helpers", "FAST corners", "Mark FAST keypoints that can be useful for frame-to-frame tracking."},
        {OperationId::OrbKeypoints, "Tracking helpers", "ORB keypoints", "Detect oriented binary features for matching or tracking."},
        {OperationId::BackgroundSubtraction, "Tracking helpers", "MOG2 background subtraction", "Highlight moving foreground using OpenCV's MOG2 background model.", true},
        {OperationId::FrameDifference, "Tracking helpers", "Frame difference", "Highlight changes between consecutive frames and draw motion boxes.", true},
        {OperationId::SparseOpticalFlow, "Tracking helpers", "Sparse LK optical flow", "Track good features between consecutive frames using pyramidal Lucas-Kanade flow.", true},
        {OperationId::OpticalFlow, "Tracking helpers", "Dense optical flow", "Draw Farneback motion vectors between consecutive frames.", true},
        {OperationId::GoodFeatures, "Tracking helpers", "Trackable corners", "Mark strong corners that are useful as tracking features."},
    };
}

const OperationInfo *MainWindow::operationInfo(OperationId id)
{
    static QVector<OperationInfo> cachedOperations = operations();
    const auto match = std::find_if(cachedOperations.cbegin(), cachedOperations.cend(), [id](const OperationInfo &operation) {
        return operation.id == id;
    });
    return match == cachedOperations.cend() ? nullptr : &(*match);
}

QVector<ParameterSpec> MainWindow::parameterSpecs(OperationId id)
{
    switch (id) {
    case OperationId::GaussianBlur:
        return {{"Kernel", 1, 51, 7}, {"Sigma x10", 0, 100, 15}};
    case OperationId::MedianBlur:
        return {{"Kernel", 1, 51, 5}};
    case OperationId::Clahe:
        return {{"Clip x10", 1, 100, 20}, {"Tile", 2, 32, 8}};
    case OperationId::BrightnessContrast:
        return {{"Contrast %", 0, 300, 120}, {"Brightness", -100, 100, 0}};
    case OperationId::GammaCorrection:
        return {{"Gamma x100", 10, 500, 80}};
    case OperationId::IlluminationCorrection:
        return {{"Bg kernel", 3, 201, 61}, {"Gain %", 20, 300, 120}};
    case OperationId::DifferenceOfGaussian:
        return {{"Small", 1, 51, 5}, {"Large", 3, 201, 41}, {"Gain x10", 1, 100, 20}};
    case OperationId::Canny:
        return {{"Low", 0, 255, 60}, {"High", 0, 255, 160}, {"Aperture", 3, 7, 3}};
    case OperationId::SobelMagnitude:
        return {{"Kernel", 1, 7, 3}, {"Scale x10", 1, 100, 10}};
    case OperationId::LaplacianEdges:
        return {{"Kernel", 1, 31, 3}, {"Scale x10", 1, 100, 10}};
    case OperationId::BinaryThreshold:
        return {{"Threshold", 0, 255, 110}, {"Max", 1, 255, 255}};
    case OperationId::AdaptiveThreshold:
        return {{"Block", 3, 99, 21}, {"C", -50, 50, 5}};
    case OperationId::HsvMask:
        return {{"Hue min", 0, 179, 0}, {"Hue max", 0, 179, 25}, {"Sat min", 0, 255, 80},
                {"Sat max", 0, 255, 255}, {"Val min", 0, 255, 80}, {"Val max", 0, 255, 255}};
    case OperationId::BrightMask:
        return {{"Value", 0, 255, 220}, {"Max sat", 0, 255, 255}, {"Open", 0, 31, 3},
                {"Close", 0, 31, 5}, {"Dilation", 0, 10, 1}};
    case OperationId::AdaptiveBrightMask:
        return {{"Bg kernel", 3, 201, 61}, {"Delta", 0, 120, 22}, {"Open", 0, 31, 3},
                {"Close", 0, 31, 5}, {"Dilation", 0, 10, 1}};
    case OperationId::Erode:
    case OperationId::Dilate:
    case OperationId::MorphOpen:
    case OperationId::MorphClose:
        return {{"Threshold", 0, 255, 110}, {"Kernel", 1, 31, 5}, {"Iterations", 1, 10, 1}};
    case OperationId::WhiteTopHat:
        return {{"Kernel", 1, 101, 31}, {"Threshold", 0, 255, 25}};
    case OperationId::BlackHat:
        return {{"Kernel", 1, 101, 31}, {"Threshold", 0, 255, 25}};
    case OperationId::Contours:
        return {{"Low", 0, 255, 50}, {"High", 0, 255, 150}, {"Min area", 0, 20000, 500}};
    case OperationId::ConnectedComponents:
        return {{"Threshold", 0, 255, 1}, {"Min area", 0, 20000, 250}, {"Max area", 0, 200000, 0},
                {"Min circ %", 0, 100, 0}, {"Dilation", 0, 10, 0}};
    case OperationId::BrightComponents:
        return {{"Value", 0, 255, 220}, {"Min area", 0, 20000, 250}, {"Max area", 0, 200000, 0},
                {"Open", 0, 31, 3}, {"Close", 0, 31, 5}, {"Dilation", 0, 10, 1}};
    case OperationId::CircularBrightBlobs:
        return {{"Value", 0, 255, 220}, {"Min area", 0, 20000, 120}, {"Max area", 0, 200000, 0},
                {"Min circ %", 0, 100, 55}, {"Open", 0, 31, 3}, {"Close", 0, 31, 5}};
    case OperationId::BestBrightCandidate:
        return {{"Value", 0, 255, 220}, {"Min area", 0, 20000, 120}, {"Max area", 0, 200000, 0},
                {"Min circ %", 0, 100, 30}, {"Open", 0, 31, 3}, {"Close", 0, 31, 5}};
    case OperationId::SimpleBlobDetector:
        return {{"Value", 0, 255, 210}, {"Min area", 0, 20000, 80}, {"Max area", 0, 200000, 0},
                {"Min circ %", 0, 100, 30}, {"Min inert %", 0, 100, 0}, {"Min conv %", 0, 100, 0}};
    case OperationId::MserRegions:
        return {{"Delta", 1, 20, 5}, {"Min area", 0, 20000, 80}, {"Max area", 1, 200000, 8000},
                {"Min value", 0, 255, 175}};
    case OperationId::FastCorners:
        return {{"Threshold", 1, 100, 24}, {"Nonmax", 0, 1, 1}, {"Max pts", 10, 2000, 500}};
    case OperationId::OrbKeypoints:
        return {{"Max pts", 10, 3000, 600}, {"Fast thr", 1, 100, 20}};
    case OperationId::HoughCircles:
        return {{"dp x10", 10, 30, 12}, {"Min dist", 1, 300, 40}, {"Canny", 1, 255, 120},
                {"Votes", 1, 255, 22}, {"Min r", 0, 200, 0}, {"Max r", 0, 300, 0}};
    case OperationId::BackgroundSubtraction:
        return {{"History", 10, 500, 120}, {"Variance", 1, 128, 16}, {"Min area", 0, 20000, 600}};
    case OperationId::RunningAverageForeground:
        return {{"Alpha x1000", 1, 500, 40}, {"Threshold", 0, 255, 28}, {"Min area", 0, 20000, 400},
                {"Dilation", 0, 10, 2}};
    case OperationId::MotionBrightMask:
        return {{"Motion", 0, 255, 25}, {"Value", 0, 255, 210}, {"Min area", 0, 20000, 250},
                {"Dilation", 0, 10, 2}, {"Open", 0, 31, 3}, {"Close", 0, 31, 5}};
    case OperationId::FrameDifference:
        return {{"Threshold", 0, 255, 32}, {"Min area", 0, 20000, 450}, {"Dilation", 0, 10, 2}};
    case OperationId::SparseOpticalFlow:
        return {{"Max pts", 10, 1000, 200}, {"Quality x1000", 1, 100, 10}, {"Distance", 1, 100, 10},
                {"Min move", 0, 50, 1}};
    case OperationId::OpticalFlow:
        return {{"Grid", 8, 64, 24}, {"Gain x10", 1, 80, 18}, {"Min mag", 0, 50, 3}};
    case OperationId::GoodFeatures:
        return {{"Max", 10, 500, 120}, {"Quality x1000", 1, 100, 10}, {"Distance", 1, 100, 12}};
    case OperationId::Original:
    case OperationId::Grayscale:
    case OperationId::EqualizeHistogram:
    case OperationId::NormalizeIntensity:
        return {};
    }

    return {};
}

QVector<int> MainWindow::defaultParameters(OperationId id)
{
    QVector<int> values;
    for (const ParameterSpec &spec : parameterSpecs(id)) {
        values.append(spec.defaultValue);
    }
    return values;
}

QString MainWindow::parameterSummary(OperationId id, const QVector<int> &parameters)
{
    const QVector<ParameterSpec> specs = parameterSpecs(id);
    if (specs.isEmpty()) {
        return {};
    }

    QStringList entries;
    for (int index = 0; index < specs.size(); ++index) {
        const ParameterSpec &spec = specs[index];
        entries.append(QString("%1 %2").arg(spec.label).arg(parameter(parameters, index, spec.defaultValue)));
    }
    return entries.join(", ");
}

QString MainWindow::pipelineStepTitle(int index, const PipelineStep &step)
{
    const OperationInfo *info = operationInfo(step.id);
    const QString name = info ? info->name : QString("Unknown operation");
    const QString summary = parameterSummary(step.id, step.parameters);
    if (summary.isEmpty()) {
        return QString("%1. %2").arg(index + 1).arg(name);
    }
    return QString("%1. %2 (%3)").arg(index + 1).arg(name, summary);
}

cv::Mat MainWindow::applyOperation(const cv::Mat &frame, PipelineStep &step)
{
    const cv::Mat bgr = toBgr(frame);
    const cv::Mat gray = toGray(frame);
    const OperationId operation = step.id;
    const QVector<int> &parameters = step.parameters;

    switch (operation) {
    case OperationId::Original:
        return frame.clone();

    case OperationId::Grayscale:
        return gray;

    case OperationId::GaussianBlur: {
        cv::Mat output;
        const int kernel = oddKernel(parameter(parameters, 0, 7), 1, 51);
        const double sigma = static_cast<double>(parameter(parameters, 1, 15)) / 10.0;
        cv::GaussianBlur(bgr, output, {kernel, kernel}, sigma);
        return output;
    }

    case OperationId::MedianBlur: {
        cv::Mat output;
        const int kernel = oddKernel(parameter(parameters, 0, 5), 1, 51);
        cv::medianBlur(bgr, output, kernel);
        return output;
    }

    case OperationId::EqualizeHistogram: {
        cv::Mat output;
        cv::equalizeHist(gray, output);
        return output;
    }

    case OperationId::Clahe: {
        cv::Mat output;
        const double clipLimit = static_cast<double>(parameter(parameters, 0, 20)) / 10.0;
        const int tileSize = std::max(2, parameter(parameters, 1, 8));
        const cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, {tileSize, tileSize});
        clahe->apply(gray, output);
        return output;
    }

    case OperationId::NormalizeIntensity: {
        cv::Mat output;
        cv::normalize(gray, output, 0, 255, cv::NORM_MINMAX, CV_8U);
        return output;
    }

    case OperationId::BrightnessContrast: {
        cv::Mat output;
        const double contrast = static_cast<double>(parameter(parameters, 0, 120)) / 100.0;
        const double brightness = static_cast<double>(parameter(parameters, 1, 0));
        bgr.convertTo(output, -1, contrast, brightness);
        return output;
    }

    case OperationId::GammaCorrection: {
        const double gamma = std::max(0.1, static_cast<double>(parameter(parameters, 0, 80)) / 100.0);
        cv::Mat lookupTable(1, 256, CV_8U);
        uchar *lookup = lookupTable.ptr();
        for (int value = 0; value < 256; ++value) {
            lookup[value] = cv::saturate_cast<uchar>(std::pow(value / 255.0, gamma) * 255.0);
        }

        cv::Mat output;
        cv::LUT(bgr, lookupTable, output);
        return output;
    }

    case OperationId::IlluminationCorrection: {
        cv::Mat grayFloat;
        cv::Mat background;
        cv::Mat corrected;
        const int kernel = oddKernel(parameter(parameters, 0, 61), 3, 201);
        const double gain = static_cast<double>(parameter(parameters, 1, 120)) / 100.0;

        gray.convertTo(grayFloat, CV_32F);
        cv::GaussianBlur(grayFloat, background, {kernel, kernel}, 0.0);
        cv::divide(grayFloat, background + 1.0F, corrected, 128.0 * gain);
        corrected.convertTo(corrected, CV_8U);
        return corrected;
    }

    case OperationId::DifferenceOfGaussian: {
        cv::Mat smallBlur;
        cv::Mat largeBlur;
        cv::Mat difference;
        cv::Mat output;
        const int smallKernel = oddKernel(parameter(parameters, 0, 5), 1, 51);
        const int largeKernel = std::max(smallKernel + 2, oddKernel(parameter(parameters, 1, 41), 3, 201));
        const double gain = static_cast<double>(parameter(parameters, 2, 20)) / 10.0;

        cv::GaussianBlur(gray, smallBlur, {smallKernel, smallKernel}, 0.0);
        cv::GaussianBlur(gray, largeBlur, {oddKernel(largeKernel, 3, 201), oddKernel(largeKernel, 3, 201)}, 0.0);
        cv::subtract(smallBlur, largeBlur, difference);
        difference.convertTo(output, CV_8U, gain, 0.0);
        return output;
    }

    case OperationId::Canny: {
        cv::Mat edges;
        const int low = parameter(parameters, 0, 60);
        const int high = std::max(low + 1, parameter(parameters, 1, 160));
        const int aperture = oddKernel(parameter(parameters, 2, 3), 3, 7);
        cv::Canny(gray, edges, low, high, aperture, true);
        return edges;
    }

    case OperationId::SobelMagnitude: {
        cv::Mat gradX;
        cv::Mat gradY;
        cv::Mat magnitude;
        cv::Mat output;
        const int kernel = oddKernel(parameter(parameters, 0, 3), 1, 7);
        const double scale = static_cast<double>(parameter(parameters, 1, 10)) / 10.0;
        cv::Sobel(gray, gradX, CV_32F, 1, 0, kernel);
        cv::Sobel(gray, gradY, CV_32F, 0, 1, kernel);
        cv::magnitude(gradX, gradY, magnitude);
        magnitude.convertTo(output, CV_8U, scale);
        return output;
    }

    case OperationId::LaplacianEdges: {
        cv::Mat laplacian;
        cv::Mat absolute;
        cv::Mat output;
        const int kernel = oddKernel(parameter(parameters, 0, 3), 1, 31);
        const double scale = static_cast<double>(parameter(parameters, 1, 10)) / 10.0;
        cv::Laplacian(gray, laplacian, CV_16S, kernel, scale);
        cv::convertScaleAbs(laplacian, output);
        return output;
    }

    case OperationId::BinaryThreshold: {
        cv::Mat output;
        cv::threshold(gray, output, parameter(parameters, 0, 110), parameter(parameters, 1, 255), cv::THRESH_BINARY);
        return output;
    }

    case OperationId::AdaptiveThreshold: {
        cv::Mat output;
        const int block = oddKernel(parameter(parameters, 0, 21), 3, 99);
        cv::adaptiveThreshold(gray, output, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, block, parameter(parameters, 1, 5));
        return output;
    }

    case OperationId::HsvMask: {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        const int hueMin = parameter(parameters, 0, 0);
        const int hueMax = parameter(parameters, 1, 25);
        const int satMin = std::min(parameter(parameters, 2, 80), parameter(parameters, 3, 255));
        const int satMax = std::max(parameter(parameters, 2, 80), parameter(parameters, 3, 255));
        const int valMin = std::min(parameter(parameters, 4, 80), parameter(parameters, 5, 255));
        const int valMax = std::max(parameter(parameters, 4, 80), parameter(parameters, 5, 255));

        cv::Mat mask;
        if (hueMin <= hueMax) {
            cv::inRange(hsv,
                        cv::Scalar(hueMin, satMin, valMin),
                        cv::Scalar(hueMax, satMax, valMax),
                        mask);
        } else {
            cv::Mat lowerWrap;
            cv::Mat upperWrap;
            cv::inRange(hsv,
                        cv::Scalar(0, satMin, valMin),
                        cv::Scalar(hueMax, satMax, valMax),
                        lowerWrap);
            cv::inRange(hsv,
                        cv::Scalar(hueMin, satMin, valMin),
                        cv::Scalar(179, satMax, valMax),
                        upperWrap);
            cv::bitwise_or(lowerWrap, upperWrap, mask);
        }

        cv::Mat output = cv::Mat::zeros(bgr.size(), bgr.type());
        bgr.copyTo(output, mask);
        return output;
    }

    case OperationId::BrightMask: {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> channels;
        cv::split(hsv, channels);

        cv::Mat valueMask;
        cv::Mat saturationMask;
        cv::threshold(channels[2], valueMask, parameter(parameters, 0, 220), 255, cv::THRESH_BINARY);
        cv::threshold(channels[1], saturationMask, parameter(parameters, 1, 255), 255, cv::THRESH_BINARY_INV);

        cv::Mat mask;
        cv::bitwise_and(valueMask, saturationMask, mask);
        cleanMask(mask,
                  parameter(parameters, 2, 3),
                  parameter(parameters, 3, 5),
                  parameter(parameters, 4, 1));
        return mask;
    }

    case OperationId::AdaptiveBrightMask: {
        cv::Mat grayFloat;
        cv::Mat background;
        cv::Mat residual;
        cv::Mat mask;
        const int kernel = oddKernel(parameter(parameters, 0, 61), 3, 201);

        gray.convertTo(grayFloat, CV_32F);
        cv::GaussianBlur(grayFloat, background, {kernel, kernel}, 0.0);
        cv::subtract(grayFloat, background, residual);
        residual.convertTo(residual, CV_8U, 1.0, 128.0);
        cv::threshold(residual, mask, 128 + parameter(parameters, 1, 22), 255, cv::THRESH_BINARY);
        cleanMask(mask,
                  parameter(parameters, 2, 3),
                  parameter(parameters, 3, 5),
                  parameter(parameters, 4, 1));
        return mask;
    }

    case OperationId::Erode:
    case OperationId::Dilate:
    case OperationId::MorphOpen:
    case OperationId::MorphClose: {
        cv::Mat mask;
        cv::threshold(gray, mask, parameter(parameters, 0, 110), 255, cv::THRESH_BINARY);

        const int kernelSize = oddKernel(parameter(parameters, 1, 5), 1, 31);
        const int iterations = parameter(parameters, 2, 1);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {kernelSize, kernelSize});

        cv::Mat output;
        if (operation == OperationId::Erode) {
            cv::erode(mask, output, kernel, {-1, -1}, iterations);
        } else if (operation == OperationId::Dilate) {
            cv::dilate(mask, output, kernel, {-1, -1}, iterations);
        } else {
            const int morphOperation = operation == OperationId::MorphOpen ? cv::MORPH_OPEN : cv::MORPH_CLOSE;
            cv::morphologyEx(mask, output, morphOperation, kernel, {-1, -1}, iterations);
        }
        return output;
    }

    case OperationId::WhiteTopHat:
    case OperationId::BlackHat: {
        cv::Mat response;
        cv::Mat output;
        const int kernelSize = oddKernel(parameter(parameters, 0, 31), 1, 101);
        const int threshold = parameter(parameters, 1, 25);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {kernelSize, kernelSize});
        cv::morphologyEx(gray, response, operation == OperationId::WhiteTopHat ? cv::MORPH_TOPHAT : cv::MORPH_BLACKHAT, kernel);
        if (threshold > 0) {
            cv::threshold(response, output, threshold, 255, cv::THRESH_BINARY);
            return output;
        }
        return response;
    }

    case OperationId::Contours: {
        cv::Mat edges;
        const int low = parameter(parameters, 0, 50);
        const int high = std::max(low + 1, parameter(parameters, 1, 150));
        cv::Canny(gray, edges, low, high);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat output = bgr.clone();
        const double minArea = static_cast<double>(parameter(parameters, 2, 500));
        for (const auto &contour : contours) {
            const double area = cv::contourArea(contour);
            if (area < minArea) {
                continue;
            }
            const cv::Rect box = cv::boundingRect(contour);
            cv::rectangle(output, box, {30, 210, 120}, 2);
            cv::drawContours(output, std::vector<std::vector<cv::Point>>{contour}, -1, {255, 180, 30}, 1);
        }
        return output;
    }

    case OperationId::ConnectedComponents: {
        cv::Mat mask = binaryMask(gray, parameter(parameters, 0, 1));
        cleanMask(mask, 0, 0, parameter(parameters, 4, 0));

        cv::Mat output = overlayMask(bgr, mask, {60, 180, 255}, 0.28);
        drawFilteredContours(output,
                             mask,
                             static_cast<double>(parameter(parameters, 1, 250)),
                             static_cast<double>(parameter(parameters, 2, 0)),
                             static_cast<double>(parameter(parameters, 3, 0)) / 100.0,
                             {40, 220, 255});
        return output;
    }

    case OperationId::BrightComponents: {
        cv::Mat mask = binaryMask(gray, parameter(parameters, 0, 220));
        cleanMask(mask,
                  parameter(parameters, 3, 3),
                  parameter(parameters, 4, 5),
                  parameter(parameters, 5, 1));

        cv::Mat output = overlayMask(bgr, mask, {40, 190, 255}, 0.32);
        drawFilteredContours(output,
                             mask,
                             static_cast<double>(parameter(parameters, 1, 250)),
                             static_cast<double>(parameter(parameters, 2, 0)),
                             0.0,
                             {30, 220, 255});
        return output;
    }

    case OperationId::CircularBrightBlobs: {
        cv::Mat mask = binaryMask(gray, parameter(parameters, 0, 220));
        cleanMask(mask,
                  parameter(parameters, 4, 3),
                  parameter(parameters, 5, 5),
                  0);

        cv::Mat output = overlayMask(bgr, mask, {40, 190, 255}, 0.32);
        drawFilteredContours(output,
                             mask,
                             static_cast<double>(parameter(parameters, 1, 120)),
                             static_cast<double>(parameter(parameters, 2, 0)),
                             static_cast<double>(parameter(parameters, 3, 55)) / 100.0,
                             {30, 220, 255});
        return output;
    }

    case OperationId::BestBrightCandidate: {
        cv::Mat mask = binaryMask(gray, parameter(parameters, 0, 220));
        cleanMask(mask,
                  parameter(parameters, 4, 3),
                  parameter(parameters, 5, 5),
                  0);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const double minArea = static_cast<double>(parameter(parameters, 1, 120));
        const double maxArea = static_cast<double>(parameter(parameters, 2, 0));
        const double minCircularity = static_cast<double>(parameter(parameters, 3, 30)) / 100.0;
        int bestIndex = -1;
        double bestScore = -1.0;

        for (int index = 0; index < static_cast<int>(contours.size()); ++index) {
            const auto &contour = contours[index];
            const double area = cv::contourArea(contour);
            if (area < minArea || (maxArea > 0.0 && area > maxArea)) {
                continue;
            }

            const double circularity = contourCircularity(contour);
            if (circularity < minCircularity) {
                continue;
            }

            cv::Mat regionMask = cv::Mat::zeros(mask.size(), CV_8U);
            cv::drawContours(regionMask, contours, index, 255, cv::FILLED);
            const double meanBrightness = cv::mean(gray, regionMask)[0];
            const double score = meanBrightness * std::sqrt(std::max(area, 1.0)) * (0.5 + circularity);
            if (score > bestScore) {
                bestScore = score;
                bestIndex = index;
            }
        }

        cv::Mat output = overlayMask(bgr, mask, {50, 150, 255}, 0.18);
        if (bestIndex >= 0) {
            const cv::Rect box = cv::boundingRect(contours[bestIndex]);
            cv::rectangle(output, box, {0, 255, 255}, 3);
            cv::drawContours(output, contours, bestIndex, {30, 230, 255}, 2);
            const cv::Moments moments = cv::moments(contours[bestIndex]);
            if (std::abs(moments.m00) > std::numeric_limits<double>::epsilon()) {
                const cv::Point center(static_cast<int>(std::round(moments.m10 / moments.m00)),
                                       static_cast<int>(std::round(moments.m01 / moments.m00)));
                cv::drawMarker(output, center, {20, 40, 255}, cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
            }
        }
        return output;
    }

    case OperationId::SimpleBlobDetector: {
        cv::SimpleBlobDetector::Params detectorParams;
        detectorParams.minThreshold = static_cast<float>(parameter(parameters, 0, 210));
        detectorParams.maxThreshold = 255.0F;
        detectorParams.thresholdStep = 10.0F;
        detectorParams.minRepeatability = 1;
        detectorParams.filterByColor = true;
        detectorParams.blobColor = 255;
        detectorParams.filterByArea = true;
        detectorParams.minArea = static_cast<float>(std::max(1, parameter(parameters, 1, 80)));
        detectorParams.maxArea = static_cast<float>(parameter(parameters, 2, 0) > 0
                                                        ? parameter(parameters, 2, 0)
                                                        : gray.rows * gray.cols);
        detectorParams.filterByCircularity = parameter(parameters, 3, 30) > 0;
        detectorParams.minCircularity = static_cast<float>(parameter(parameters, 3, 30)) / 100.0F;
        detectorParams.filterByInertia = parameter(parameters, 4, 0) > 0;
        detectorParams.minInertiaRatio = static_cast<float>(parameter(parameters, 4, 0)) / 100.0F;
        detectorParams.filterByConvexity = parameter(parameters, 5, 0) > 0;
        detectorParams.minConvexity = static_cast<float>(parameter(parameters, 5, 0)) / 100.0F;

        std::vector<cv::KeyPoint> keypoints;
        cv::SimpleBlobDetector::create(detectorParams)->detect(gray, keypoints);

        cv::Mat output;
        cv::drawKeypoints(bgr, keypoints, output, {30, 220, 255}, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        return output;
    }

    case OperationId::MserRegions: {
        const int delta = parameter(parameters, 0, 5);
        const int minArea = std::max(1, parameter(parameters, 1, 80));
        const int maxArea = std::max(minArea + 1, parameter(parameters, 2, 8000));
        const int minValue = parameter(parameters, 3, 175);

        std::vector<std::vector<cv::Point>> regions;
        std::vector<cv::Rect> boxes;
        cv::MSER::create(delta, minArea, maxArea)->detectRegions(gray, regions, boxes);

        cv::Mat output = bgr.clone();
        const cv::Rect imageBounds(0, 0, gray.cols, gray.rows);
        for (int index = 0; index < static_cast<int>(boxes.size()); ++index) {
            const cv::Rect box = boxes[index] & imageBounds;
            if (box.empty() || cv::mean(gray(box))[0] < minValue) {
                continue;
            }
            cv::rectangle(output, box, {30, 220, 255}, 2);
            cv::drawContours(output, regions, index, {255, 190, 40}, 1);
        }
        return output;
    }

    case OperationId::FastCorners: {
        std::vector<cv::KeyPoint> keypoints;
        const int threshold = parameter(parameters, 0, 24);
        const bool nonmax = parameter(parameters, 1, 1) != 0;
        const int maxPoints = parameter(parameters, 2, 500);

        cv::FastFeatureDetector::create(threshold, nonmax)->detect(gray, keypoints);
        if (static_cast<int>(keypoints.size()) > maxPoints) {
            std::nth_element(keypoints.begin(), keypoints.begin() + maxPoints, keypoints.end(), [](const cv::KeyPoint &left, const cv::KeyPoint &right) {
                return left.response > right.response;
            });
            keypoints.resize(maxPoints);
        }

        cv::Mat output;
        cv::drawKeypoints(bgr, keypoints, output, {30, 220, 255}, cv::DrawMatchesFlags::DEFAULT);
        return output;
    }

    case OperationId::OrbKeypoints: {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        const int maxPoints = parameter(parameters, 0, 600);
        const int fastThreshold = parameter(parameters, 1, 20);

        const cv::Ptr<cv::ORB> detector = cv::ORB::create(maxPoints);
        detector->setFastThreshold(fastThreshold);
        detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

        cv::Mat output;
        cv::drawKeypoints(bgr, keypoints, output, {30, 220, 255}, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        return output;
    }

    case OperationId::HoughCircles: {
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, {9, 9}, 2.0);

        std::vector<cv::Vec3f> circles;
        const double dp = static_cast<double>(parameter(parameters, 0, 12)) / 10.0;
        const double minDistance = static_cast<double>(parameter(parameters, 1, 40));
        const double canny = static_cast<double>(parameter(parameters, 2, 120));
        const double votes = static_cast<double>(parameter(parameters, 3, 22));
        const int minRadius = parameter(parameters, 4, 0);
        const int maxRadius = parameter(parameters, 5, 0);
        cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, dp, minDistance, canny, votes, minRadius, maxRadius);

        cv::Mat output = bgr.clone();
        for (const cv::Vec3f &circle : circles) {
            const cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
            const int radius = cvRound(circle[2]);
            cv::circle(output, center, radius, {30, 220, 255}, 2, cv::LINE_AA);
            cv::drawMarker(output, center, {20, 40, 255}, cv::MARKER_CROSS, 14, 2, cv::LINE_AA);
        }
        return output;
    }

    case OperationId::BackgroundSubtraction: {
        const int history = parameter(parameters, 0, 120);
        const int threshold = parameter(parameters, 1, 16);
        const double minArea = static_cast<double>(parameter(parameters, 2, 600));

        if (!step.backgroundSubtractor || step.backgroundHistory != history || step.backgroundThreshold != threshold) {
            step.backgroundSubtractor = cv::createBackgroundSubtractorMOG2(history, static_cast<double>(threshold), true);
            step.backgroundHistory = history;
            step.backgroundThreshold = threshold;
        }

        cv::Mat foregroundMask;
        step.backgroundSubtractor->apply(bgr, foregroundMask);
        cv::threshold(foregroundMask, foregroundMask, 200, 255, cv::THRESH_BINARY);

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5});
        cv::morphologyEx(foregroundMask, foregroundMask, cv::MORPH_OPEN, kernel);
        cv::dilate(foregroundMask, foregroundMask, kernel, {-1, -1}, 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(foregroundMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat output = bgr.clone();
        cv::Mat highlight(output.size(), output.type(), cv::Scalar(25, 25, 180));
        highlight.copyTo(output, foregroundMask);
        cv::addWeighted(bgr, 0.68, output, 0.32, 0.0, output);

        for (const auto &contour : contours) {
            if (cv::contourArea(contour) >= minArea) {
                cv::rectangle(output, cv::boundingRect(contour), {40, 220, 255}, 2);
            }
        }
        return output;
    }

    case OperationId::RunningAverageForeground: {
        cv::Mat grayFloat;
        gray.convertTo(grayFloat, CV_32F);
        cv::Mat output = bgr.clone();

        if (step.runningAverage.empty()) {
            step.runningAverage = grayFloat.clone();
            drawWaitingLabel(output, "Need next frame");
            return output;
        }

        cv::Mat background;
        cv::Mat delta;
        cv::Mat mask;
        step.runningAverage.convertTo(background, CV_8U);
        cv::absdiff(gray, background, delta);
        cv::threshold(delta, mask, parameter(parameters, 1, 28), 255, cv::THRESH_BINARY);
        cleanMask(mask, 3, 5, parameter(parameters, 3, 2));

        output = overlayMask(bgr, mask, {40, 190, 255}, 0.32);
        drawFilteredContours(output,
                             mask,
                             static_cast<double>(parameter(parameters, 2, 400)),
                             0.0,
                             0.0,
                             {30, 220, 255});
        const double alpha = static_cast<double>(parameter(parameters, 0, 40)) / 1000.0;
        cv::accumulateWeighted(grayFloat, step.runningAverage, alpha);
        return output;
    }

    case OperationId::MotionBrightMask: {
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, {5, 5}, 0);
        cv::Mat output = bgr.clone();

        if (step.previousGrayFrame.empty()) {
            step.previousGrayFrame = blurred.clone();
            drawWaitingLabel(output, "Need next frame");
            return output;
        }

        cv::Mat motion;
        cv::Mat bright;
        cv::Mat mask;
        cv::absdiff(step.previousGrayFrame, blurred, motion);
        cv::threshold(motion, motion, parameter(parameters, 0, 25), 255, cv::THRESH_BINARY);
        cv::threshold(gray, bright, parameter(parameters, 1, 210), 255, cv::THRESH_BINARY);
        cv::bitwise_and(motion, bright, mask);
        cleanMask(mask,
                  parameter(parameters, 4, 3),
                  parameter(parameters, 5, 5),
                  parameter(parameters, 3, 2));

        output = overlayMask(bgr, mask, {40, 190, 255}, 0.35);
        drawFilteredContours(output,
                             mask,
                             static_cast<double>(parameter(parameters, 2, 250)),
                             0.0,
                             0.0,
                             {30, 220, 255});

        step.previousGrayFrame = blurred.clone();
        return output;
    }

    case OperationId::FrameDifference: {
        cv::Mat blurred;
        cv::GaussianBlur(gray, blurred, {5, 5}, 0);
        cv::Mat output = bgr.clone();

        if (step.previousGrayFrame.empty()) {
            step.previousGrayFrame = blurred.clone();
            drawWaitingLabel(output, "Need next frame");
            return output;
        }

        cv::Mat delta;
        cv::absdiff(step.previousGrayFrame, blurred, delta);
        cv::threshold(delta, delta, parameter(parameters, 0, 32), 255, cv::THRESH_BINARY);

        const int dilation = parameter(parameters, 2, 2);
        if (dilation > 0) {
            const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
            cv::dilate(delta, delta, kernel, {-1, -1}, dilation);
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(delta, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat highlight(output.size(), output.type(), cv::Scalar(30, 30, 210));
        highlight.copyTo(output, delta);
        cv::addWeighted(bgr, 0.72, output, 0.28, 0.0, output);

        const double minArea = static_cast<double>(parameter(parameters, 1, 450));
        for (const auto &contour : contours) {
            if (cv::contourArea(contour) >= minArea) {
                cv::rectangle(output, cv::boundingRect(contour), {40, 230, 255}, 2);
            }
        }

        step.previousGrayFrame = blurred.clone();
        return output;
    }

    case OperationId::SparseOpticalFlow: {
        cv::Mat output = bgr.clone();
        const int maxPoints = parameter(parameters, 0, 200);
        const double quality = static_cast<double>(parameter(parameters, 1, 10)) / 1000.0;
        const double minDistance = static_cast<double>(parameter(parameters, 2, 10));
        const double minMove = static_cast<double>(parameter(parameters, 3, 1));

        if (step.previousGrayFrame.empty() || step.previousPoints.empty()) {
            cv::goodFeaturesToTrack(gray, step.previousPoints, maxPoints, quality, minDistance);
            step.previousGrayFrame = gray.clone();
            for (const cv::Point2f &point : step.previousPoints) {
                cv::circle(output, {cvRound(point.x), cvRound(point.y)}, 3, {30, 220, 255}, -1, cv::LINE_AA);
            }
            drawWaitingLabel(output, "Need next frame");
            return output;
        }

        std::vector<cv::Point2f> nextPoints;
        std::vector<uchar> status;
        std::vector<float> errors;
        cv::calcOpticalFlowPyrLK(step.previousGrayFrame, gray, step.previousPoints, nextPoints, status, errors);

        std::vector<cv::Point2f> goodPoints;
        goodPoints.reserve(nextPoints.size());
        for (int index = 0; index < static_cast<int>(nextPoints.size()); ++index) {
            if (!status[index]) {
                continue;
            }

            const cv::Point2f previous = step.previousPoints[index];
            const cv::Point2f current = nextPoints[index];
            if (current.x < 0.0F || current.y < 0.0F || current.x >= gray.cols || current.y >= gray.rows) {
                continue;
            }

            const double movement = std::hypot(current.x - previous.x, current.y - previous.y);
            if (movement >= minMove) {
                cv::arrowedLine(output,
                                {cvRound(previous.x), cvRound(previous.y)},
                                {cvRound(current.x), cvRound(current.y)},
                                {40, 220, 255},
                                1,
                                cv::LINE_AA,
                0,
                0.25);
            }
            cv::circle(output, {cvRound(current.x), cvRound(current.y)}, 3, {20, 40, 255}, -1, cv::LINE_AA);
            goodPoints.push_back(current);
        }

        if (static_cast<int>(goodPoints.size()) < std::max(10, maxPoints / 4)) {
            cv::goodFeaturesToTrack(gray, goodPoints, maxPoints, quality, minDistance);
        }

        step.previousPoints = goodPoints;
        step.previousGrayFrame = gray.clone();
        return output;
    }

    case OperationId::OpticalFlow: {
        cv::Mat output = bgr.clone();
        if (step.previousGrayFrame.empty()) {
            step.previousGrayFrame = gray.clone();
            drawWaitingLabel(output, "Need next frame");
            return output;
        }

        cv::Mat flow;
        cv::calcOpticalFlowFarneback(step.previousGrayFrame, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        const int grid = parameter(parameters, 0, 24);
        const double gain = static_cast<double>(parameter(parameters, 1, 18)) / 10.0;
        const double minMagnitude = static_cast<double>(parameter(parameters, 2, 3));
        for (int y = grid / 2; y < output.rows; y += grid) {
            for (int x = grid / 2; x < output.cols; x += grid) {
                const cv::Point2f vector = flow.at<cv::Point2f>(y, x);
                const double magnitude = std::hypot(vector.x, vector.y);
                if (magnitude < minMagnitude) {
                    continue;
                }
                const cv::Point start{x, y};
                const cv::Point end{
                    static_cast<int>(std::round(x + vector.x * gain)),
                    static_cast<int>(std::round(y + vector.y * gain))};
                cv::arrowedLine(output, start, end, {40, 220, 255}, 1, cv::LINE_AA, 0, 0.32);
            }
        }

        step.previousGrayFrame = gray.clone();
        return output;
    }

    case OperationId::GoodFeatures: {
        std::vector<cv::Point2f> corners;
        const int maxCorners = parameter(parameters, 0, 120);
        const double quality = static_cast<double>(parameter(parameters, 1, 10)) / 1000.0;
        const double minDistance = static_cast<double>(parameter(parameters, 2, 12));
        cv::goodFeaturesToTrack(gray, corners, maxCorners, quality, minDistance);

        cv::Mat output = bgr.clone();
        for (const cv::Point2f &corner : corners) {
            const cv::Point center(cvRound(corner.x), cvRound(corner.y));
            cv::circle(output, center, 4, {30, 220, 255}, 2, cv::LINE_AA);
            cv::circle(output, center, 1, {20, 30, 40}, -1, cv::LINE_AA);
        }
        return output;
    }
    }

    return frame.clone();
}

int MainWindow::parameter(const QVector<int> &parameters, int index, int fallback)
{
    if (index < 0 || index >= parameters.size()) {
        return fallback;
    }
    return parameters[index];
}

bool MainWindow::hasSelectedPipelineStep() const
{
    return selectedStepIndex_ >= 0 && selectedStepIndex_ < pipeline_.size();
}
