# set source class path
set (SRC_FILES_PATH "../../src")

# Find OpenCV Package
find_package(OpenCV REQUIRED)

# Find Tesseract Package
find_package(Tesseract REQUIRED)

set (LIBRARIES ${OpenCV_LIBS} libtesseract )

set(Main ${SRC_FILES_PATH}/main.cpp
        )

add_executable(SpellCheck ${Main})
include_directories (SpellCheck ${OpenCV_INCLUDE_DIRS})
target_link_libraries (SpellCheck ${LIBRARIES})
