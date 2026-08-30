#pragma once

#include <QImage>
#include <QMainWindow>
#include <QString>
#include <QTimer>
#include <QVector>

#include <array>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>

class QGroupBox;
class QGridLayout;
class QLabel;
class QLineEdit;
class QListWidget;
class QPushButton;
class QScrollArea;
class QSlider;
class QSpinBox;
class QTreeWidget;
class QTreeWidgetItem;
class QWidget;

class ImageView;

enum class OperationId {
    Original = 0,
    Grayscale,
    GaussianBlur,
    MedianBlur,
    EqualizeHistogram,
    Clahe,
    NormalizeIntensity,
    BrightnessContrast,
    GammaCorrection,
    Canny,
    SobelMagnitude,
    LaplacianEdges,
    BinaryThreshold,
    AdaptiveThreshold,
    HsvMask,
    BrightMask,
    AdaptiveBrightMask,
    IlluminationCorrection,
    DifferenceOfGaussian,
    Erode,
    Dilate,
    MorphOpen,
    MorphClose,
    WhiteTopHat,
    BlackHat,
    Contours,
    ConnectedComponents,
    BrightComponents,
    CircularBrightBlobs,
    BestBrightCandidate,
    SimpleBlobDetector,
    MserRegions,
    FastCorners,
    OrbKeypoints,
    HoughCircles,
    BackgroundSubtraction,
    RunningAverageForeground,
    MotionBrightMask,
    FrameDifference,
    SparseOpticalFlow,
    OpticalFlow,
    GoodFeatures
};

struct OperationInfo {
    OperationId id;
    QString category;
    QString name;
    QString description;
    bool temporal = false;
};

struct ParameterSpec {
    QString label;
    int minimum = 0;
    int maximum = 100;
    int defaultValue = 0;
};

struct ParameterWidgets {
    QWidget *row = nullptr;
    QLabel *label = nullptr;
    QSlider *slider = nullptr;
    QSpinBox *spinBox = nullptr;
};

struct PipelineStep {
    OperationId id = OperationId::Original;
    QVector<int> parameters;
    cv::Mat previousGrayFrame;
    cv::Mat runningAverage;
    std::vector<cv::Point2f> previousPoints;
    cv::Ptr<cv::BackgroundSubtractorMOG2> backgroundSubtractor;
    int backgroundHistory = 0;
    int backgroundThreshold = 0;
};

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);

private:
    static constexpr int ParameterCount = 6;

    void buildUi();
    void buildOperationTree();
    void configureParameters(OperationId id, const QVector<int> &values = {});
    void filterOperations(const QString &filter);
    void loadMedia(const QString &path);
    void resetMedia();
    void resetTemporalState();
    void resetTemporalStateFrom(int firstStep);
    void resetPreviewState();
    void resetStepState(PipelineStep &step);
    void selectOperation(OperationId id);
    void selectPipelineStep(int row);
    void addSelectedOperationToChain();
    void removeSelectedPipelineStep();
    void moveSelectedPipelineStep(int direction);
    void clearPipeline();
    void updatePipelineList();
    void updatePipelineControls();
    void storeParameterValue(int index, int value);
    void processCurrentFrame();
    void updateStageImages(const QVector<QString> &titles, const QVector<cv::Mat> &frames);
    void rebuildStageViews(const QVector<QString> &titles);
    void updateMediaStatus();
    void updatePlaybackControls();
    void seekVideo(int frameNumber);
    void advanceVideo();
    void setPlaying(bool playing);
    void stepVideo(int frameDelta);

    [[nodiscard]] static QVector<OperationInfo> operations();
    [[nodiscard]] static const OperationInfo *operationInfo(OperationId id);
    [[nodiscard]] static QVector<ParameterSpec> parameterSpecs(OperationId id);
    [[nodiscard]] static QVector<int> defaultParameters(OperationId id);
    [[nodiscard]] static QString parameterSummary(OperationId id, const QVector<int> &parameters);
    [[nodiscard]] static QString pipelineStepTitle(int index, const PipelineStep &step);
    [[nodiscard]] cv::Mat applyOperation(const cv::Mat &frame, PipelineStep &step);
    [[nodiscard]] static int parameter(const QVector<int> &parameters, int index, int fallback);
    [[nodiscard]] bool hasSelectedPipelineStep() const;

    QTreeWidget *operationTree_ = nullptr;
    QLineEdit *operationFilter_ = nullptr;
    QListWidget *pipelineList_ = nullptr;
    QLabel *mediaStatusLabel_ = nullptr;
    QLabel *operationDescriptionLabel_ = nullptr;
    QLabel *frameLabel_ = nullptr;
    QGroupBox *parameterGroup_ = nullptr;
    QPushButton *openButton_ = nullptr;
    QPushButton *addStepButton_ = nullptr;
    QPushButton *removeStepButton_ = nullptr;
    QPushButton *moveStepUpButton_ = nullptr;
    QPushButton *moveStepDownButton_ = nullptr;
    QPushButton *clearChainButton_ = nullptr;
    QPushButton *playPauseButton_ = nullptr;
    QPushButton *stopButton_ = nullptr;
    QPushButton *previousFrameButton_ = nullptr;
    QPushButton *nextFrameButton_ = nullptr;
    QSlider *timelineSlider_ = nullptr;
    QScrollArea *stageScrollArea_ = nullptr;
    QWidget *stageContainer_ = nullptr;
    QGridLayout *stageGridLayout_ = nullptr;

    std::array<ParameterWidgets, ParameterCount> parameterWidgets_;
    QVector<int> parameterValues_ = QVector<int>(ParameterCount, 0);
    QVector<PipelineStep> pipeline_;
    PipelineStep previewStep_;
    QVector<ImageView *> stageViews_;
    QVector<QString> stageTitles_;

    QTimer frameTimer_;
    cv::VideoCapture capture_;
    cv::Mat currentFrame_;

    QString currentPath_;
    OperationId selectedOperation_ = OperationId::Original;
    bool isVideo_ = false;
    bool isPlaying_ = false;
    bool configuringParameters_ = false;
    bool updatingPipelineList_ = false;
    int frameIndex_ = 0;
    int frameCount_ = 0;
    int selectedStepIndex_ = -1;
    double framesPerSecond_ = 30.0;
};
