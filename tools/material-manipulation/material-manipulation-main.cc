/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2016 Zdravko Velinov
*
*   Permission is hereby granted, free of charge, to any person obtaining a copy
*   of this software and associated documentation files (the "Software"), to deal
*   in the Software without restriction, including without limitation the rights
*   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*   copies of the Software, and to permit persons to whom the Software is
*   furnished to do so, subject to the following conditions:
*
*   The above copyright notice and this permission notice shall be included in
*   all copies or substantial portions of the Software.
*
*   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*   THE SOFTWARE.
*/


#include <QMessageBox>
#include <QMainWindow>
#include <QKeyEvent>
#include <QAbstractNativeEventFilter>
#ifdef _WIN32
#   include <qpa/qplatformnativeinterface.h>
#endif

#include "ui_material-manipulation.h"

#include "tempest/utils/parse-command-line.hh"
#include "material-manipulation.hh"
#include "tempest/graphics/os-window.hh"
#include "tempest/utils/timer.hh"
#include "tempest/utils/video-encode.hh"

const uint32_t ImageWidth = 720;
const uint32_t ImageHeight = 480;

typedef decltype(Tempest::PreferredSystem().Library) LibraryType;

#define RECORD_FILE "record.bin"
#define VIDEO_FILE "fabric-manipulation.ivf"

//#define RECORD_CAPS

class RecordFilter;

int64_t g_GlobalTime = 0;

const uint32_t FPS = 30;

class MaterialManipulationWindow: public QMainWindow
{
	Q_OBJECT

	Ui::MaterialManipulator                         m_UI;

    QString                                         m_PreviousFile;
    QString                                         m_NextFile;
    uint                                            m_CurrentFile;

    Tempest::PreferredBackend                       m_Backend;
    Tempest::PreferredShaderCompiler                m_ShaderCompiler;

    MaterialManipulation                            m_MaterialManipulation;

    Tempest::TimeQuery                              m_Timer;
    int64_t                                         m_PlaybackStart = 0,
                                                    m_RecordStart = 0;

    std::unique_ptr<Tempest::VPXVideoEncoder>       m_VideoEncoder;
    Tempest::OSWindow                               m_WindowNativeHandle;

    std::unique_ptr<EventRecord[]>                  m_EventRecords;
    size_t                                          m_EventCount = 0,
                                                    m_ParsedEvent;
    int64_t                                         m_NextFrameMouseX,
                                                    m_NextFrameMouseY,
                                                    m_PreviousFrameMouseX,
                                                    m_PreviousFrameMouseY;
    int64_t                                         m_PreviousFrameTime = 0;
    enum class NextFrameStatus
    {
        Uninitialized,
        MoveMouse,
        Stay
    }                                               m_NextFrameStatus = NextFrameStatus::Stay;
public:
	MaterialManipulationWindow(float yaw, float roll)
        :   m_MaterialManipulation(yaw, roll)
	{
		m_UI.setupUi(this);
    }

    bool init(LibraryType& library, uint32_t flags)
    {
        m_UI.MaterialView->attach(&m_Backend);

        auto status = library.initGraphicsLibrary();
        if(!status)
            return false;
        m_Backend.init();

        QWindow* window = this->windowHandle();
        if(!window)
        {
            QMessageBox::critical(this, "Error", "Failed to take screenshot");
            return false;
        }
        
    #ifdef _WIN32
        QPlatformNativeInterface* iface = QGuiApplication::platformNativeInterface();
        m_WindowNativeHandle = static_cast<Tempest::OSWindow>(iface->nativeResourceForWindow(QByteArrayLiteral("handle"), window));
    #else
        TGE_ASSERT(false, "implement on linux");
    #endif

        m_MaterialManipulation.setDisableRecord(true);

        status = m_MaterialManipulation.init(ImageWidth, ImageHeight, flags, &m_UI.MaterialView->getTempestWindow(),  &m_Backend, &m_ShaderCompiler);
        if(!status)
        {
            return false;
        }

        auto diffuse_color = Tempest::ConvertLinearToSRGB(m_MaterialManipulation.getDiffuseColor());
        QColor diffuse_qt_color(int(diffuse_color.x*255.0f), int(diffuse_color.y*255.0f), int(diffuse_color.z*255.0f));
        m_UI.DiffuseColorPicker->setColor(diffuse_qt_color);

        auto specular_color = Tempest::ConvertLinearToSRGB(m_MaterialManipulation.getSpecularColor());
        QColor spec_qt_color(int(specular_color.x*255.0f), int(specular_color.y*255.0f), int(specular_color.z*255.0f));
        m_UI.SpecularColorPicker->setColor(spec_qt_color);

        if(flags & MATERIAL_MANIPULATION_GUI_RECORD)
        {
            m_VideoEncoder = decltype(m_VideoEncoder)(new Tempest::VPXVideoEncoder);

            std::unique_ptr<Tempest::Texture> test_shot(Tempest::CaptureScreenshot(m_WindowNativeHandle, true));
            
            if(!test_shot)
            {
                QMessageBox::critical(this, "Error", "Failed to capture screenshot");
                return false;
            }
            
            auto& hdr = test_shot->getHeader();
            
            Tempest::VideoInfo video_info;
            video_info.Width = hdr.Width;
            video_info.Height = hdr.Height;
            video_info.FileName = VIDEO_FILE;
            video_info.Bitrate = 50000;
            video_info.FPS = FPS;

            m_MaterialManipulation.setAreaLightSampleCount(32);

            auto status = m_VideoEncoder->openStream(video_info);
            if(!status)
            {
                QMessageBox::critical(this, "Error", "Failed to open video stream: " VIDEO_FILE);
                return false;
            }

            if(!startPlayback())
                return false;
        }
        else
        {
            g_GlobalTime = m_Timer.time();
        }

        m_RecordStart = m_Timer.time() + 5000000ULL;

        return true;
    }

    void takeScreenshot();

private slots:
    void on_MaterialView_rendering()
    {
        auto cur_time = m_Timer.time();
        if(!m_VideoEncoder)
        {
            g_GlobalTime = cur_time;
        }

        m_MaterialManipulation.setActive(isActiveWindow());

        if(!m_MaterialManipulation.run(false))
            close();

        if(cur_time > m_RecordStart)
        {
            if(m_VideoEncoder)
            {
                std::unique_ptr<Tempest::Texture> shot(Tempest::CaptureScreenshot(m_WindowNativeHandle, true));
                if(shot)
                {
                    m_VideoEncoder->submitFrame(*shot);
                    g_GlobalTime += 1000000LL/FPS;
                }
                else
                {
                    QMessageBox::critical(this, "Error", "Failed to capture screenshot");
                    m_VideoEncoder.reset();
                }

                if(m_EventCount == m_ParsedEvent)
                {
                    this->close();
                    return;
                }
            }
            
            auto render_time = g_GlobalTime - m_PlaybackStart;
            for(; m_ParsedEvent < m_EventCount; ++m_ParsedEvent)
            {
                auto& evt = m_EventRecords[m_ParsedEvent];
      
                if(evt.Timestamp >= render_time)
                {
                    switch(m_NextFrameStatus)
                    {
                    case NextFrameStatus::Uninitialized:
                    {
                        m_NextFrameStatus = NextFrameStatus::Stay;
                        for(size_t idx = m_ParsedEvent; idx < m_EventCount; ++idx)
                        {
                            auto& sub_evt = m_EventRecords[idx];
                            if(evt.Timestamp != sub_evt.Timestamp)
                                break;

                            if(sub_evt.Event.Type != Tempest::WindowEventType::MouseMoved)
                                continue;

                            m_NextFrameMouseX = sub_evt.Event.MouseMoved.MouseX;
                            m_NextFrameMouseY = sub_evt.Event.MouseMoved.MouseY;
                            m_NextFrameStatus = NextFrameStatus::MoveMouse;
                        }
                    } break;
                    case NextFrameStatus::MoveMouse:
                    {
                        Tempest::WindowSystemEvent gen_evt;
                        gen_evt.Type = Tempest::WindowEventType::MouseMoved;
                        gen_evt.MouseMoved.MouseX = m_PreviousFrameMouseX + (m_NextFrameMouseX - m_PreviousFrameMouseX)*(render_time - m_PreviousFrameTime)/(evt.Timestamp - m_PreviousFrameTime);
                        gen_evt.MouseMoved.MouseY = m_PreviousFrameMouseY + (m_NextFrameMouseY - m_PreviousFrameMouseY)*(render_time - m_PreviousFrameTime)/(evt.Timestamp - m_PreviousFrameTime);
                        Tempest::GenerateEvent(evt.Event);
                    } break;
                    }
                    break;
                }
                Tempest::GenerateEvent(evt.Event);
                m_PreviousFrameTime = evt.Timestamp;
                if(evt.Event.Type == Tempest::WindowEventType::MouseMoved)
                {
                    m_PreviousFrameMouseX = evt.Event.MouseMoved.MouseX;
                    m_PreviousFrameMouseY = evt.Event.MouseMoved.MouseY;
                    m_PreviousFrameTime = evt.Timestamp;
                }
                m_NextFrameStatus = NextFrameStatus::Uninitialized;
            }
        }
    }

    void on_DiffuseSlider_sliderMoved(int value)
    {
        m_UI.DiffuseSpinBox->blockSignals(true);
            float value_f = powf(10.0f, value*1e-3f);
            m_UI.DiffuseSpinBox->setValue(value_f);
            m_MaterialManipulation.setDiffuseMultiplier(value_f);
        m_UI.DiffuseSpinBox->blockSignals(false);
    }

    void on_DiffuseSpinBox_valueChanged(double d)
    {
        m_UI.DiffuseSlider->blockSignals(true);
            m_UI.DiffuseSlider->setValue(1000*log10(d));
            m_MaterialManipulation.setDiffuseMultiplier(static_cast<float>(d));
        m_UI.DiffuseSlider->blockSignals(false);
    }

    void on_SpecularSlider_sliderMoved(int value)
    {
        m_UI.SpecularSpinBox->blockSignals(true);
            float value_f = powf(10.0f, value*1e-3f);
            m_UI.SpecularSpinBox->setValue(value_f);
            m_MaterialManipulation.setSpecularMultiplier(value_f);
        m_UI.SpecularSpinBox->blockSignals(false);
    }

    void on_SpecularSpinBox_valueChanged(double d)
    {
        m_UI.SpecularSlider->blockSignals(true);
            m_UI.SpecularSlider->setValue(1000*log10(d));
            m_MaterialManipulation.setSpecularMultiplier(static_cast<float>(d));
        m_UI.SpecularSlider->blockSignals(false);
    }

    void on_AnisotropySlider_sliderMoved(int value)
    {
        m_UI.AnisotropySpinBox->blockSignals(true);
            float value_f = value*1e-3f;
            m_UI.AnisotropySpinBox->setValue(value_f);
            m_MaterialManipulation.setAnisotropyModifier(value_f);
        m_UI.AnisotropySpinBox->blockSignals(false);
    }

    void on_AnisotropySpinBox_valueChanged(double d)
    {
        m_UI.AnisotropySlider->blockSignals(true);
            m_UI.AnisotropySlider->setValue(d*1e3f);
            m_MaterialManipulation.setAnisotropyModifier(static_cast<float>(d));
        m_UI.AnisotropySlider->blockSignals(false);
    }

    void on_SmoothnessSlider_sliderMoved(int value)
    {
        m_UI.SmoothnessSpinBox->blockSignals(true);
            float value_f = value*1e-3f;
            m_UI.SmoothnessSpinBox->setValue(value_f);
            m_MaterialManipulation.setSmoothnessModifier(value_f);
        m_UI.SmoothnessSpinBox->blockSignals(false);
    }

    void on_SmoothnessSpinBox_valueChanged(double d)
    {
        m_UI.SmoothnessSlider->blockSignals(true);
            m_UI.SmoothnessSlider->setValue(d*1e3f);
            m_MaterialManipulation.setSmoothnessModifier(static_cast<float>(d));
        m_UI.SmoothnessSlider->blockSignals(false);
    }

    void on_DiffuseColorPicker_colorChanged(const QColor& color)
    {
        float scale = 1.0f/255.0f;
        m_MaterialManipulation.setDiffuseColor(Tempest::ToVector4(Tempest::ConvertSRGBToLinear(Tempest::Vector3{ color.red()*scale, color.green()*scale, color.blue()*scale })));
    }

    void on_SpecularColorPicker_colorChanged(const QColor& color)
    {
        float scale = 1.0f/255.0f;
        m_MaterialManipulation.setSpecularColor(Tempest::ToVector4(Tempest::ConvertSRGBToLinear(Tempest::Vector3{ color.red()*scale, color.green()*scale, color.blue()*scale })));
    }

    void on_actionTake_Screenshot_triggered(bool checked = false)
    {
        Tempest::TexturePtr texture(Tempest::CaptureScreenshot(m_WindowNativeHandle, true));

        std::stringstream ss;
		ss << "screenshot.png";
		uint32_t counter = 0;
		for(;;)
		{
			if(!Tempest::System::Exists(ss.str()))
				break;
			ss.str("");
			ss << "screenshot_" << counter++ << ".png";
		}

        auto status = Tempest::SaveImage(texture->getHeader(), texture->getData(), Tempest::Path(ss.str()));
        if(!status)
        {
            QMessageBox::critical(this, "Error", "Failed to take screenshot");
            return;
        }
    }

    void on_actionTake_Screenshot_Without_GUI_triggered(bool checked = false)
    {
        m_MaterialManipulation.takeScreenshot();
    }

    virtual void keyReleaseEvent(QKeyEvent* evt) override
    {
        if(m_VideoEncoder)
            return;

        switch(evt->key())
        {
        case Qt::Key_P:
        {
            startPlayback();

            m_PlaybackStart = m_Timer.time();
        } break;
        }
    }

private:
    bool startPlayback()
    {
        m_ParsedEvent = m_EventCount = 0;
        std::fstream fs(RECORD_FILE, std::ios::in|std::ios::binary);

        auto start = fs.tellg();
        fs.seekg(0, std::ios::end);
        size_t size = fs.tellg() - start;
        fs.seekg(0, std::ios::beg);

        if(size % sizeof(m_EventRecords[0]))
        {
            QMessageBox::critical(this, "Error", "Invalid event record file: ", RECORD_FILE);
            return false;
        }

        m_EventCount = size/sizeof(m_EventRecords[0]);

        m_EventRecords = decltype(m_EventRecords)(new EventRecord[m_EventCount]);
        fs.read(reinterpret_cast<char*>(m_EventRecords.get()), size);
        if(!fs)
        {
            fs.close();
            return false;
        }

        return true;
    }
};

Tempest::MouseButtonId TranslateQtMouseButtons(Qt::MouseButton button)
{
    switch(button)
    {
    case Qt::LeftButton: return Tempest::MouseButtonId::LeftButton;
    case Qt::MidButton: return Tempest::MouseButtonId::MiddleButton;
    case Qt::RightButton: return Tempest::MouseButtonId::RightButton; 
    case Qt::ExtraButton1: return Tempest::MouseButtonId::ExtraButton1;
    case Qt::ExtraButton2: return Tempest::MouseButtonId::ExtraButton2;
    }

    return Tempest::MouseButtonId::Unknown;
}

// Bypass all shitty APIs
#if defined(_WIN32) && defined(RECORD_CAPS)
namespace EventHooks
{
const int EventCount = 10*1024*1024;
static std::mutex s_WriteProtect;
static uint64_t s_RecordStart = 0;
static int32_t s_PrevMouseX = 0, s_PrevMouseY = 0;
HHOOK s_NextMouseHook, s_NextKeyboardHook;
std::unique_ptr<EventRecord[]> s_EventBuffer;
std::atomic_int s_EventCounter;
bool s_Record = false;

static void PushMouseMoveEvent()
{
    int32_t mouse_x, mouse_y;
    Tempest::GetMousePosition(&mouse_x, &mouse_y);
    if(mouse_x == s_PrevMouseX && mouse_y == s_PrevMouseY)
        return;

    auto idx = s_EventCounter.fetch_add(1);
    if(idx >= EventCount)
        return;

    EventRecord& wevent = s_EventBuffer[idx];
    wevent.Timestamp = g_GlobalTime - s_RecordStart;

    wevent.Event.Type = Tempest::WindowEventType::MouseMoved;
    wevent.Event.MouseMoved.MouseX = mouse_x;
    wevent.Event.MouseMoved.MouseY = mouse_y;
    wevent.Event.MouseMoved.MouseDeltaX = mouse_x - s_PrevMouseX;
    wevent.Event.MouseMoved.MouseDeltaY = mouse_y - s_PrevMouseY;

    s_PrevMouseX = mouse_x;
    s_PrevMouseY = mouse_y;
}

static void PushMouseButtonPressedEvent(Tempest::WindowEventType mouse_evt_type, Tempest::MouseButtonId button)
{
    auto status = (uint32_t)(mouse_evt_type == Tempest::WindowEventType::MouseButtonPressed) << (uint32_t)button;

    PushMouseMoveEvent();

    auto idx = s_EventCounter.fetch_add(1);
    if(idx >= EventCount)
        return;

    EventRecord& wevent = s_EventBuffer[idx];
    wevent.Timestamp = g_GlobalTime - s_RecordStart;
    wevent.Event.Type = mouse_evt_type;
    wevent.Event.MouseButton = button;
}



LRESULT CALLBACK LowLevelMouseHook(int ncode, WPARAM wparam, LPARAM lparam)
{

    if(ncode >= 0 && s_Record)
    {
        switch(wparam)
        {
        case WM_MOUSEMOVE: PushMouseMoveEvent(); break;
        case WM_LBUTTONDOWN: PushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonPressed, Tempest::MouseButtonId::LeftButton); break;
        case WM_LBUTTONUP: PushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonReleased, Tempest::MouseButtonId::LeftButton); break;
        case WM_MBUTTONDOWN: PushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonPressed, Tempest::MouseButtonId::MiddleButton); break;
        case WM_MBUTTONUP: PushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonReleased, Tempest::MouseButtonId::MiddleButton); break;
        case WM_RBUTTONDOWN: PushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonPressed, Tempest::MouseButtonId::RightButton); break;
        case WM_RBUTTONUP: PushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonReleased, Tempest::MouseButtonId::RightButton); break;
        // case WM_XBUTTONDOWN: pushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonPressed, TranslateExtraButton(wParam), &info->EventQueue); break;
        // case WM_XBUTTONUP: pushMouseButtonPressedEvent(Tempest::WindowEventType::MouseButtonReleased, TranslateExtraButton(wParam), &info->EventQueue); break;
        }
    }

    return CallNextHookEx(s_NextMouseHook, ncode, wparam, lparam);
}


LRESULT CALLBACK LowLevelKeyboardHook(int ncode, WPARAM wparam, LPARAM lparam)
{
    if(ncode >= 0)
    {
        switch(wparam)
        {
        case WM_KEYUP:
        {
            if(((LPKBDLLHOOKSTRUCT)lparam)->vkCode != 'U')
                break;

            //m_Message = message;

            if(s_Record)
            {
                s_Record = false;

                Tempest::Log(Tempest::LogLevel::Info, "ENDED RECORDING EVENTS");
                std::fstream output_stream(RECORD_FILE, std::ios::binary|std::ios::out);
                if(!output_stream)
                {
                    Tempest::Log(Tempest::LogLevel::Error, "Failed to open file for recording");
                    break;
                }

                output_stream.write(reinterpret_cast<const char*>(s_EventBuffer.get()), sizeof(s_EventBuffer[0])*std::min(EventCount, s_EventCounter.load()));
                s_EventCounter.store(0);
            }
            else
            {
                s_Record = true;
                s_RecordStart = g_GlobalTime;

                Tempest::GetMousePosition(&s_PrevMouseX, &s_PrevMouseY);
            }
        } break;
        }
    }

    return CallNextHookEx(s_NextKeyboardHook, ncode, wparam, lparam);
}

static void Install()
{
    s_NextMouseHook = SetWindowsHookEx(WH_MOUSE_LL, &EventHooks::LowLevelMouseHook, GetModuleHandle(NULL), 0);
    s_NextKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, &EventHooks::LowLevelKeyboardHook, GetModuleHandle(NULL), 0);

    s_EventBuffer = decltype(s_EventBuffer)(new EventRecord[EventCount]);
}
};

#endif

#undef Bool
#undef Void
#include "material-manipulation-main.moc"

int TempestMain(int argc, char** argv)
{
    Tempest::CommandLineOptsParser parser("material-manipulation", false);
    parser.createOption('P', "playback-and-record", "Playback recorded set of events and record them on video", false);
    parser.createOption('t', "test-rotation", "Specify rotation period in ms of the test rotations", true, "0");
    parser.createOption('g', "gui", "Enable editing GUI", false);
    parser.createOption('p', "print-profile", "Enable profile printing", false);
    parser.createOption('a', "area-light", "Use point light source", false);
    parser.createOption('L', "low-spec", "Use lower level of detail", false);
    parser.createOption('Y', "camera-yaw", "Specify camera yaw", true);
    parser.createOption('R', "camera-roll", "Specify camera roll", true);

    auto status = parser.parse(argc, argv);
    if(!status)
    {
        return EXIT_FAILURE;
    }

    #ifdef ENABLE_PROFILER
    if(parser.isSet("print-profile"))
    {
        Tempest::Profiler::setPrintMode(true);
    }    
    #endif

    float yaw = 0.0f;
    if(parser.isSet("camera-yaw"))
    {
        yaw = Tempest::ToRadians(parser.extract<float>("camera-yaw"));
    }
    
    float roll = (30.0f * Tempest::MathPi) / 180.0f;
    if(parser.isSet("camera-roll"))
    {
        roll = Tempest::ToRadians(parser.extract<float>("camera-roll"));
    }

    uint32_t flags = 0;
    if(parser.isSet("area-light"))
    {
        flags |= MATERIAL_MANIPULATION_AREA_LIGHT;
    }

    if(parser.isSet("low-spec"))
    {
        flags |= MATERIAL_MANIPULATION_LOW_SPEC;
    }

    if(parser.isSet("gui"))
    {
        QApplication app(argc, argv);

        LibraryType library;
        auto status = library.initDeviceContextLibrary();
        if(!status)
            return false;

    #if defined(_WIN32) && defined(RECORD_CAPS)
        EventHooks::Install();
    #endif

        MaterialManipulationWindow wnd(yaw, roll);
        
        if(parser.isSet("playback-and-record"))
        {
            flags |= MATERIAL_MANIPULATION_GUI_RECORD;
        }

        wnd.init(library, flags);

        wnd.show();

        return app.exec();
    }
    else
    {
        MaterialManipulation material_manipulation;

        material_manipulation.setPlaybackAndRecord(parser.isSet("playback-and-record"));
        material_manipulation.setTestRotationPeriod(parser.extract<uint64_t>("test-rotation")*1000UL);

        status = material_manipulation.init(ImageWidth, ImageHeight, flags);
        if(!status)
        {
            return EXIT_FAILURE;
        }

        while(material_manipulation.run())
            ;
    }

    return EXIT_SUCCESS;
}
