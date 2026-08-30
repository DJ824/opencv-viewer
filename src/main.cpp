#include "MainWindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QApplication::setApplicationName("OpenCV Operation Viewer");
    QApplication::setOrganizationName("opencv-viewer");

    MainWindow window;
    window.show();

    return QApplication::exec();
}
