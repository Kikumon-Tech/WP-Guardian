cmake_minimum_required(VERSION 3.15)
project(WP_Guardian)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0148 NEW)
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

# vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "Your vcpkg directory" CACHE STRING "")
# GLFW, OpenGL, OpenCV, GLEW, CPR
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(cpr REQUIRED)

# ImGui
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui)
set(IMGUI_SRC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# 実行ファイル
add_executable(WP_Guardian
    main.cpp
    ${IMGUI_SRC}
    app.rc
)

if(MSVC)
    target_link_options(WP_Guardian PRIVATE "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
endif()

# include
target_include_directories(WP_Guardian PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${OpenCV_INCLUDE_DIRS}
)
# link
target_link_libraries(WP_Guardian PRIVATE
    glfw
    OpenGL::GL
    ${OpenCV_LIBS}
    GLEW::GLEW
    cpr::cpr
)