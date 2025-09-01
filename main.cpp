#define NOMINMAX // min/maxマクロの競合を回避

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <opencv2/opencv.hpp>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <cpr/cpr.h>
#include <vector>
#include <algorithm> // std::min/std::max
#include <tuple>
#include <windows.h>
#include <commdlg.h>
#include <string>
#include <fstream> // C++のファイルストリームを使用

void setWindowIconFromExe(GLFWwindow* window) {
    HWND hwnd = glfwGetWin32Window(window);
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);

    HICON hIconLarge = nullptr;
    HICON hIconSmall = nullptr;
    ExtractIconExA(exePath, 0, &hIconLarge, &hIconSmall, 1);

    if (hIconLarge) {
        HICON prev = (HICON)SendMessage(hwnd, WM_SETICON, ICON_BIG, (LPARAM)hIconLarge);
        if (prev) DestroyIcon(prev);
    }

    if (hIconSmall) {
        HICON prev = (HICON)SendMessage(hwnd, WM_SETICON, ICON_SMALL, (LPARAM)hIconSmall);
        if (prev) DestroyIcon(prev);
    }
}


using namespace cv;

std::atomic<bool> abort_fetch{false};

std::atomic<bool> stopThread{false};

static bool showOriginal = true;
static bool showRealtime = true;
static bool showDiff = true;
static bool showSettings = true;
static bool showInfo = false;

static int tile_x = 1818;
static int tile_y = 806;
static int pixel_x = 989;
static int pixel_y = 359;

static std::string szFile;
char szFileBuffer[MAX_PATH] = {0};

// INIファイルの絶対パスを格納するグローバル変数
std::string iniPath;

// アプリケーション設定をINIファイルに保存する関数
void SaveAppSettings() {
    std::ofstream ofs(iniPath);
    if (!ofs.is_open()) {
        std::cerr << "INIファイルを保存できませんでした: " << iniPath << std::endl;
        return;
    }

    // ウィンドウの表示状態
    ofs << "showOriginal=" << showOriginal << std::endl;
    ofs << "showRealtime=" << showRealtime << std::endl;
    ofs << "showDiff=" << showDiff << std::endl;
    ofs << "showSettings=" << showSettings << std::endl;
    ofs << "showInfo=" << showInfo << std::endl;

    // 位置系
    ofs << "tile_x=" << tile_x << std::endl;
    ofs << "tile_y=" << tile_y << std::endl;
    ofs << "pixel_x=" << pixel_x << std::endl;
    ofs << "pixel_y=" << pixel_y << std::endl;

    // パス
    ofs << "path=" << szFile << std::endl;

    ofs.close();
}

// アプリケーション設定をINIファイルから読み込む関数
void LoadAppSettings() {
    std::ifstream ifs(iniPath);
    if (!ifs.is_open()) {
        std::cerr << "INIファイルが見つかりません: " << iniPath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);

        try {
            if (key == "showOriginal") showOriginal = (std::stoi(val) != 0);
            else if (key == "showRealtime") showRealtime = (std::stoi(val) != 0);
            else if (key == "showDiff") showDiff = (std::stoi(val) != 0);
            else if (key == "showSettings") showSettings = (std::stoi(val) != 0);
            else if (key == "showInfo") showInfo = (std::stoi(val) != 0);
            else if (key == "tile_x") tile_x = std::stoi(val);
            else if (key == "tile_y") tile_y = std::stoi(val);
            else if (key == "pixel_x") pixel_x = std::stoi(val);
            else if (key == "pixel_y") pixel_y = std::stoi(val);
            else if (key == "path") szFile = val;
        } catch (const std::exception& e) {
            std::cerr << "INI file parse error: " << e.what() << std::endl;
        }
    }
    ifs.close();
}


cv::Mat applyAlphaMask(const cv::Mat& src, const cv::Mat& mask) {
    if (src.empty() || mask.empty()) return cv::Mat();
    CV_Assert(src.size() == mask.size());
    CV_Assert(src.type() == CV_8UC4 && mask.type() == CV_8UC4);

    cv::Mat dst = src.clone();
    for (int y = 0; y < src.rows; ++y) {
        const cv::Vec4b* srcRow = src.ptr<cv::Vec4b>(y);
        const cv::Vec4b* maskRow = mask.ptr<cv::Vec4b>(y);
        cv::Vec4b* dstRow = dst.ptr<cv::Vec4b>(y);
        for (int x = 0; x < src.cols; ++x) {
            uchar srcA = srcRow[x][3] ? 255 : 0;
            uchar maskA = maskRow[x][3] ? 255 : 0;
            dstRow[x][3] = srcA & maskA;
        }
    }
    return dst;
}

void ensureBGRA(cv::Mat& img){
    if (img.empty()) return;
    if (img.channels()==3) cv::cvtColor(img,img,COLOR_BGR2BGRA);
}



std::tuple<cv::Mat, int, int> imageDifferenceSafe(const cv::Mat& img1, const cv::Mat& img2) {
    if (img1.empty() || img2.empty()) return {cv::Mat(), 0, 0};

    int w = std::min(img1.cols, img2.cols);
    int h = std::min(img1.rows, img2.rows);
    if (w <= 0 || h <= 0) return {cv::Mat(), 0, 0};
    
    cv::Mat diffImage = cv::Mat::zeros(img1.size(), CV_8UC4);
    cv::Rect roi(0, 0, w, h);
    cv::Mat roi1 = img1(roi);
    cv::Mat roi2 = img2(roi);

    // 差分（RGB絶対値）
    cv::Mat rgbDiff;
    if (roi1.channels() >= 3 && roi2.channels() >= 3) {
        std::vector<cv::Mat> ch1, ch2, chDiff(3);
        cv::split(roi1, ch1);
        cv::split(roi2, ch2);
        for (int i = 0; i < 3; ++i) cv::absdiff(ch1[i], ch2[i], chDiff[i]);
        cv::merge(chDiff, rgbDiff);
    } else {
        cv::absdiff(roi1, roi2, rgbDiff);
    }

    // アルファチャンネル
    cv::Mat alpha;
    if (roi1.channels() == 4) cv::extractChannel(roi1, alpha, 3);
    else alpha = cv::Mat::ones(roi.size(), CV_8U) * 255;

    // RGBAにマージ
    cv::Mat merged;
    std::vector<cv::Mat> ch;
    cv::split(rgbDiff, ch);
    ch.push_back(alpha);
    cv::merge(ch, merged);



    merged.copyTo(diffImage(roi));

    // マスク作成：元画像の不透明ピクセル
    cv::Mat opaqueMask = alpha > 0;

    // 「間違いピクセル」のカウント
    int changedPixels = 0;
    for (int y = 0; y < h; ++y) {
        const cv::Vec4b* p1 = roi1.ptr<cv::Vec4b>(y);
        const cv::Vec4b* p2 = roi2.ptr<cv::Vec4b>(y);
        const uchar* maskPtr = opaqueMask.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) {
            if (maskPtr[x]) {
                if (p1[x] != p2[x]) changedPixels++;
            }
        }
    }

    int totalOpaquePixels = cv::countNonZero(opaqueMask);
    return {diffImage, totalOpaquePixels, changedPixels};
}

cpr::Response cancellable_fetch(const std::string& url,const cpr::Header& headers){
    try{
        auto future = cpr::GetAsync(cpr::Url{url},headers,cpr::Timeout{5000});
        while(!abort_fetch && future.wait_for(std::chrono::milliseconds(50))!=std::future_status::ready){}
        if(abort_fetch){ try{future.get();}catch(...){}} 
        return future.get();
    }catch(...){ return cpr::Response{};}
}

cv::Mat fetch_tiles_and_crop_cpp(
    int tile_x, int tile_y, int x_in_tile, int y_in_tile,
    int ref_width, int ref_height, int TILE_SIZE = 1000)
{
    int end_x = x_in_tile + ref_width;
    int end_y = y_in_tile + ref_height;
    int tile_x_end = tile_x + (end_x / TILE_SIZE);
    int tile_y_end = tile_y + (end_y / TILE_SIZE);

    int canvas_w = (tile_x_end - tile_x + 1) * TILE_SIZE;
    int canvas_h = (tile_y_end - tile_y + 1) * TILE_SIZE;
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC4, cv::Scalar(0,0,0,0));

    cpr::Header headers = {
        {"User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        {"Accept", "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"},
        {"Referer", "https://www.google.com/"}
    };

    int tile_index = 0;
    for (int ty = tile_y; ty <= tile_y_end; ++ty) {
        for (int tx = tile_x; tx <= tile_x_end; ++tx) {
            if (abort_fetch) return cv::Mat();
            auto now = std::chrono::system_clock::now();
            auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
            std::string url = "https://backend.wplace.live/files/s0/tiles/" + std::to_string(tx) + "/" + std::to_string(ty) + ".png" + "?t=" + std::to_string(epoch);
            cpr::Response r = cancellable_fetch(url, headers);
            if (abort_fetch) return cv::Mat();
            if (r.status_code != 200 || r.text.empty()) return cv::Mat();

            std::vector<uchar> data(r.text.begin(), r.text.end());
            cv::Mat img = cv::imdecode(data, cv::IMREAD_UNCHANGED);
            if (img.empty()) return cv::Mat();

            int ty_rel = tile_index / (tile_x_end - tile_x + 1);
            int tx_rel = tile_index % (tile_x_end - tile_x + 1);
            int oy = ty_rel * TILE_SIZE;
            int ox = tx_rel * TILE_SIZE;

            cv::Rect roi_dst(ox, oy, img.cols, img.rows);
            cv::Rect canvas_rect(0, 0, canvas.cols, canvas.rows);
            roi_dst &= canvas_rect;

            if (roi_dst.width > 0 && roi_dst.height > 0) {
                cv::Rect roi_src(0, 0, roi_dst.width, roi_dst.height);
                img(roi_src).copyTo(canvas(roi_dst));
            }
            tile_index++;
        }
    }

    cv::Rect roi(x_in_tile, y_in_tile, ref_width, ref_height);
    cv::Rect canvas_rect(0, 0, canvas.cols, canvas.rows);
    roi &= canvas_rect;
    if (roi.width <= 0 || roi.height <= 0) return cv::Mat();
    return canvas(roi).clone();
}

GLuint matToTexture(const cv::Mat& mat){
    if(mat.empty()) return 0;
    GLuint texID;
    glGenTextures(1,&texID);
    glBindTexture(GL_TEXTURE_2D,texID);
    GLenum format = (mat.channels()==3)? GL_BGR : GL_BGRA;
    GLint internalFormat = (mat.channels()==3)? GL_RGB8 : GL_RGBA8;
    glTexImage2D(GL_TEXTURE_2D,0,internalFormat,mat.cols,mat.rows,0,format,GL_UNSIGNED_BYTE,mat.data);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D,0);
    return texID;
}

int main(){
    // 実行ファイルのディレクトリを取得し、INIファイルの絶対パスを作成
    char pathBuf[MAX_PATH];
    GetModuleFileNameA(NULL, pathBuf, MAX_PATH);
    std::string fullPath(pathBuf);
    size_t lastSlash = fullPath.find_last_of("\\/");
    std::string appDir = fullPath.substr(0, lastSlash + 1);
    iniPath = appDir + "app_settings.ini";

    std::string imguiIniPath = appDir + "imgui.ini";

    if(!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1200,800,"WP-Guardian",nullptr,nullptr);
    if(!window){ glfwTerminate(); return -1; }
    setWindowIconFromExe(window);
    glfwMakeContextCurrent(window);
    glewExperimental = GL_TRUE;
    glewInit();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.IniFilename = "imgui.ini";
    ImGui_ImplGlfw_InitForOpenGL(window,true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
    ImGui::StyleColorsDark();

    ImFontConfig font_cfg{};
    font_cfg.FontDataOwnedByAtlas = true;

    ImFont* font = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/meiryo.ttc", 18.0f, &font_cfg, io.Fonts->GetGlyphRangesJapanese());
    ImFont* bigFont = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/meiryo.ttc", 25.0f, &font_cfg, io.Fonts->GetGlyphRangesJapanese());
    if (!font) std::cerr << "フォント読み込み失敗！\n";
    if (!bigFont) std::cerr << "フォント読み込み失敗！\n";
    
    // アプリ起動時に設定を読み込む
    LoadAppSettings();
    // szFileBufferにszFileの値をコピーして、ImGuiの初期値を設定
    strncpy(szFileBuffer, szFile.c_str(), MAX_PATH);

    cv::Mat originalImg = cv::imread(szFile, cv::IMREAD_UNCHANGED);
    if (originalImg.empty()) {
        originalImg = cv::Mat(1,1,CV_8UC4,cv::Scalar(0,0,0,0));
    }
    ensureBGRA(originalImg);
    GLuint originalTexID = matToTexture(originalImg);
    int width   = originalImg.cols;
    int height = originalImg.rows;


    cv::Mat realtimeImg(height,width,CV_8UC4,cv::Scalar(0,0,0,0));
    GLuint realtimeTexID = matToTexture(realtimeImg);

    cv::Mat emptyDiff(height, width, CV_8UC4, cv::Scalar(0,0,0,0));
    GLuint diffTexID = matToTexture(emptyDiff);

    static int tmpTile_x  = tile_x;
    static int tmpTile_y  = tile_y;
    static int tmpPixel_x = pixel_x;
    static int tmpPixel_y = pixel_y;

    double diffPercent = 0.0;
    int totalOpaquePixels = 0;
    int changedPixels = 0;

    std::mutex imgMutex;
    std::condition_variable cv_newFrame;
    bool newFrameReady = false;


    std::thread updateThread([&](){
        while(!stopThread){
            cv::Mat newImg = fetch_tiles_and_crop_cpp(tile_x,tile_y,pixel_x,pixel_y,width,height);
            if (!newImg.empty()) {
                newImg = applyAlphaMask(newImg, originalImg);
                std::lock_guard<std::mutex> lock(imgMutex);
                realtimeImg = newImg.clone();
                auto [diffImg, totalOpaque, changed] = imageDifferenceSafe(originalImg, realtimeImg);
                diffPercent = (totalOpaque > 0) ? (double)changed / totalOpaque * 100.0 : 0.0;
                totalOpaquePixels = totalOpaque;
                changedPixels = changed;

                emptyDiff = diffImg.clone();
                newFrameReady = true;
                cv_newFrame.notify_one();
            }
            for(int i=0;i<10 && !stopThread;i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    glfwSetWindowCloseCallback(window, [](GLFWwindow* win) {
        glfwSetWindowShouldClose(win, GLFW_TRUE);
        abort_fetch = true;
        stopThread = true;
    });


    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::DockSpaceOverViewport(viewport->ID,viewport,ImGuiDockNodeFlags_PassthruCentralNode);

        float scale = 1.0f;
        ImVec2 imageSize;

        if (ImGui::BeginMainMenuBar())
        {
            if (ImGui::BeginMenu("ウィンドウ"))
            {
                ImGui::MenuItem("オリジナル画像", nullptr, &showOriginal);
                ImGui::MenuItem("リアルタイム画像", nullptr, &showRealtime);
                ImGui::MenuItem("差分画像", nullptr, &showDiff);
                ImGui::MenuItem("設定", nullptr, &showSettings);
                ImGui::MenuItem("情報", nullptr, &showInfo);
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }

        if (showOriginal){
            ImGui::Begin("オリジナル画像", &showOriginal);
            float scale = std::min((float)ImGui::GetContentRegionAvail().x / width,(float)ImGui::GetContentRegionAvail().y / height);
            ImVec2 imageSize(width * scale, height * scale);
            ImGui::Image((void*)(intptr_t)originalTexID, imageSize);
            ImGui::End();
        }

        if (showRealtime) {
            ImGui::Begin("リアルタイム画像", &showRealtime);
            scale = std::min((float)ImGui::GetContentRegionAvail().x / width,(float)ImGui::GetContentRegionAvail().y / height);
            imageSize = ImVec2(width * scale, height * scale);
            ImGui::Image((void*)(intptr_t)realtimeTexID, imageSize);
            ImGui::End();
        }

        if (showDiff)  {
            ImGui::Begin("差分画像", &showDiff);
            scale = std::min((float)ImGui::GetContentRegionAvail().x / width,(float)ImGui::GetContentRegionAvail().y / height);
            imageSize = ImVec2(width * scale, height * scale);
            ImGui::Image((void*)(intptr_t)diffTexID, imageSize);
            ImGui::End();
        }
        if (showSettings) {
            ImGui::Begin("設定", &showSettings);
            float windowWidth   = ImGui::GetContentRegionAvail().x;
            float itemWidth     = ImGui::GetContentRegionAvail().x*0.9f;
            float xPos          = (windowWidth - itemWidth) * 0.5f;

            ImGui::SetCursorPosX(xPos);
            ImGui::PushFont(bigFont);
            ImGui::Text("座標");
            ImGui::PopFont();
            ImGui::PushItemWidth(itemWidth);
            ImGui::SetCursorPosX(xPos);
            ImGui::InputInt("Tl X", &tmpTile_x);
            ImGui::SetCursorPosX(xPos);
            ImGui::InputInt("Tl Y", &tmpTile_y);
            ImGui::SetCursorPosX(xPos);
            ImGui::InputInt("Px X", &tmpPixel_x);
            ImGui::SetCursorPosX(xPos);
            ImGui::InputInt("Px Y", &tmpPixel_y);
            ImGui::PopItemWidth();
            ImGui::Spacing();

            ImGui::SetCursorPosX(xPos);
            ImGui::PushFont(bigFont);
            ImGui::Text("使用画像パス");
            ImGui::PopFont();
            ImGui::PushItemWidth(itemWidth - 80); // ボタン幅80px確保
            ImGui::SetCursorPosX(xPos);
            // szFileBufferを使用
            ImGui::InputText("##画像保存先", szFileBuffer, MAX_PATH);
            ImGui::PopItemWidth();

            ImGui::SameLine();
            ImGui::SetCursorPosX(xPos + itemWidth - 80); // ボタンの開始位置
            if (ImGui::Button("参照", ImVec2(80,0))) {
                OPENFILENAME ofn;
                ZeroMemory(&ofn, sizeof(ofn));
                ofn.lStructSize = sizeof(ofn);

                HWND hwnd = glfwGetWin32Window(window);
                ofn.hwndOwner = hwnd;
                
                // ImGuiバッファを使用
                ofn.lpstrFile = szFileBuffer;
                ofn.nMaxFile = sizeof(szFileBuffer);
                ofn.lpstrFilter = "PNG Files\0*.png\0JPEG Files\0*.jpg;*.jpeg\0All Files\0*.*\0";
                ofn.nFilterIndex = 1;
                ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

                if (GetOpenFileName(&ofn)) {
                    // ダイアログでパスが選択されたら、std::string変数にコピー
                    szFile = szFileBuffer;
                }
            }
            ImGui::Spacing();

            ImGui::SetCursorPosX(xPos);
            if (ImGui::Button("更新", ImVec2(itemWidth, 0))) {
                std::lock_guard<std::mutex> lock(imgMutex);
                
                // ImGuiバッファからszFileに最新のパスをコピー
                szFile = szFileBuffer;

                if (!szFile.empty()) {
                    cv::Mat newImg = cv::imread(szFile, cv::IMREAD_UNCHANGED);
                    if (!newImg.empty()) {
                        originalImg = newImg.clone();
                        ensureBGRA(originalImg);
                        if (originalTexID) glDeleteTextures(1, &originalTexID);
                        originalTexID = matToTexture(originalImg);
                        width   = originalImg.cols;
                        height = originalImg.rows;
                        realtimeImg = cv::Mat(height, width, CV_8UC4, cv::Scalar(0,0,0,0));
                        emptyDiff   = cv::Mat(height, width, CV_8UC4, cv::Scalar(0,0,0,0));

                        if(realtimeTexID) glDeleteTextures(1, &realtimeTexID);
                        realtimeTexID = matToTexture(realtimeImg);

                        if(diffTexID) glDeleteTextures(1, &diffTexID);
                        diffTexID = matToTexture(emptyDiff);

                    }
                }

                tile_x  = tmpTile_x;
                tile_y  = tmpTile_y;
                pixel_x = tmpPixel_x;
                pixel_y = tmpPixel_y;

                abort_fetch = true;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                abort_fetch = false;
            }
            ImGui::End();
        }

        if (showInfo)  {
            ImGui::Begin("情報", &showInfo);
            ImGui::PushFont(bigFont);
            ImGui::Text("差分率: %.2f%%", diffPercent);
            ImGui::Text("%d / %d", changedPixels, totalOpaquePixels);
            ImGui::PopFont();
            ImGui::End();
        }

        std::unique_lock<std::mutex> lock(imgMutex);
        if (cv_newFrame.wait_for(lock, std::chrono::milliseconds(1), [&]{ return newFrameReady; })) {
            if (!realtimeImg.empty() && realtimeImg.cols == width && realtimeImg.rows == height){
                GLenum format = (realtimeImg.channels()==3)? GL_BGR : GL_BGRA;
                glBindTexture(GL_TEXTURE_2D,realtimeTexID);
                glTexSubImage2D(GL_TEXTURE_2D,0,0,0,realtimeImg.cols,realtimeImg.rows,format,GL_UNSIGNED_BYTE,realtimeImg.data);
                glBindTexture(GL_TEXTURE_2D,0);
            }
            if(!emptyDiff.empty() && emptyDiff.cols == width && emptyDiff.rows == height){
                glBindTexture(GL_TEXTURE_2D, diffTexID);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, emptyDiff.cols, emptyDiff.rows, GL_BGRA, GL_UNSIGNED_BYTE, emptyDiff.data);
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            newFrameReady = false;
        }

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window,&display_w,&display_h);
        glViewport(0,0,display_w,display_h);
        glClearColor(0.1f,0.1f,0.1f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }

    stopThread = true;
    if(updateThread.joinable()) updateThread.join();

    glDeleteTextures(1, &originalTexID);
    glDeleteTextures(1, &realtimeTexID);
    glDeleteTextures(1, &diffTexID);

    SaveAppSettings();
    ImGui::SaveIniSettingsToDisk(imguiIniPath.c_str());
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
