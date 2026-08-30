# OpenCV Operation Viewer

A cross-platform Qt 6 desktop app for loading images and videos, browsing OpenCV operations, and comparing the original input against the processed output. The operation set is oriented toward prototyping object tracking pipelines.

## Features

- Load common image formats and video files.
- Scrollable stage viewer that shows the input plus the output after every operation in the chain.
- Operation browser with filter/search.
- Editable operation chain with add, single-step remove, reorder, and clear controls.
- Chain rows display saved parameter values and update when a selected step's sliders or spin boxes change.
- Adjustable parameters for contrast normalization, illumination correction, thresholding, HSV masks, bright masks, morphology, contours, connected components, blob detection, MSER, Hough circles, frame differencing, running-average foreground extraction, MOG2 background subtraction, sparse/dense optical flow, and corner/keypoint detection.
- Video playback, frame stepping, and timeline scrubbing.

## Bright object tracking starter chains

- `CLAHE` -> `White top-hat` -> `Bright components`
- `Divide by local background` -> `Adaptive bright mask` -> `Circular bright blobs`
- `Motion-gated bright mask` -> `Best bright candidate`
- `Difference of Gaussian` -> `Bright mask` -> `Simple blob detector`

## Dependencies

- CMake 3.24 or newer
- A C++20 compiler
- Qt 6.5 or newer with Widgets
- OpenCV 4 with `core`, `imgproc`, `imgcodecs`, `video`, `videoio`, and `features2d`

### Linux

Ubuntu/Debian:

```bash
sudo apt install cmake g++ qt6-base-dev libopencv-dev
```

Fedora:

```bash
sudo dnf install cmake gcc-c++ qt6-qtbase-devel opencv-devel
```

### Windows

The recommended Windows setup is Visual Studio 2022 plus vcpkg. This keeps Qt, OpenCV, headers, libraries, and DLL search paths consistent for CMake.

#### Option A: Visual Studio 2022 + vcpkg

Install prerequisites:

- Visual Studio 2022 with the `Desktop development with C++` workload.
- Git for Windows.
- CMake, either from Visual Studio, the Qt installer, or cmake.org.
- Ninja is optional, but useful for faster command-line builds.

Install vcpkg:

```powershell
cd C:\dev
git clone https://github.com/microsoft/vcpkg.git
cd C:\dev\vcpkg
.\bootstrap-vcpkg.bat
setx VCPKG_ROOT C:\dev\vcpkg
setx PATH "$env:PATH;C:\dev\vcpkg"
```

Open a new `Developer PowerShell for VS 2022`, then install dependencies:

```powershell
vcpkg install qtbase opencv4:x64-windows
```

Configure and build:

```powershell
cd C:\path\to\opencv-viewer
cmake -S . -B build-win `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows

cmake --build build-win --config Release
```

Run:

```powershell
.\build-win\Release\opencv_viewer.exe
```

If video files do not open, install OpenCV with FFmpeg support and rebuild:

```powershell
vcpkg install "opencv4[ffmpeg]:x64-windows"
cmake --build build-win --config Release
```

#### Option B: Ninja command-line build

Use this if you prefer a single-config build directory:

```powershell
cd C:\path\to\opencv-viewer
cmake -S . -B build-win-ninja `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows

cmake --build build-win-ninja
.\build-win-ninja\opencv_viewer.exe
```

#### Option C: Qt Creator or CLion

For Qt Creator, open this folder as a CMake project and configure the kit to use MSVC 2022 x64. Add this CMake option:

```text
-DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
```

For CLion, go to `Settings > Build, Execution, Deployment > CMake` and add the same option to the Windows profile. If vcpkg is somewhere else, use that absolute path instead.

#### Packaging a Windows folder

For sharing the app with another Windows machine, build `Release`, then collect Qt runtime files with `windeployqt`:

```powershell
cd C:\path\to\opencv-viewer
mkdir dist
copy .\build-win\Release\opencv_viewer.exe .\dist\
windeployqt .\dist\opencv_viewer.exe
```

`windeployqt` handles Qt DLLs and plugins. OpenCV DLLs still need to be available through `PATH` or copied from vcpkg's installed binary folder, usually:

```powershell
C:\dev\vcpkg\installed\x64-windows\bin
```

For a portable folder, copy the required `opencv_*.dll` files from that folder into `dist`.

#### Common Windows fixes

- `Could not find Qt6Config.cmake`: pass the vcpkg toolchain file, or set `CMAKE_PREFIX_PATH` to your Qt install path.
- `Could not find OpenCVConfig.cmake`: pass the vcpkg toolchain file, or set `OpenCV_DIR` to the folder containing `OpenCVConfig.cmake`.
- `opencv_viewer.exe` starts from the IDE but not from Explorer: Qt or OpenCV DLLs are missing from `PATH`; run `windeployqt` and copy the OpenCV DLLs into the same folder as the `.exe`.
- Debug/Release mismatch errors: build dependencies and the app for the same architecture and configuration, preferably `x64-windows` and `Release`.

Official references:

- Microsoft vcpkg CMake integration: https://learn.microsoft.com/en-us/vcpkg/users/buildsystems/cmake-integration
- Qt Windows deployment and `windeployqt`: https://doc.qt.io/qt-6/windows-deployment.html

## Build

```bash
cmake -S . -B build
cmake --build build
```

Run the resulting `opencv_viewer` executable from the build directory.
