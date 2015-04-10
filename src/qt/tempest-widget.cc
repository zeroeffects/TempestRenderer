/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2014 Zdravko Velinov
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

#include "tempest/qt/tempest-widget.hh"
#include "tempest/graphics/os-window.hh"

#include "tempest/graphics/opengl-backend/gl-window.hh"
#include "tempest/graphics/opengl-backend/gl-backend.hh"

#include <thread>
#include <QLayout>

namespace Tempest
{
extern Tempest::OSWindowSystem& GetWindowSystem()
{
    static Tempest::OSWindowSystem singleton;
    return singleton;
}

extern OSWindowSystem& GetWindowSystem();

#ifndef NDEBUG
std::thread::id DebugThreadID;
#endif

TempestWidget::TempestWidget(QWidget* parent)
    :   QWidget(parent, 0)
{
    this->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

    auto* layout = new QHBoxLayout(this);

    WindowDescription wdesc;
    wdesc.Width = 800;
    wdesc.Height = 600;
    wdesc.Title = "Tempest Widget";
    m_Window.init(GetWindowSystem(), reinterpret_cast<Tempest::OSWindow>(this->winId()), wdesc);
    auto* embed = QWidget::createWindowContainer(QWindow::fromWinId((WId)m_Window.getWindowId()), this, Qt::MSWindowsOwnDC);
    embed->setMinimumSize(QSize(200, 200));
    embed->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    embed->setFocusPolicy(Qt::ClickFocus);
    
    layout->addWidget(embed);

    startTimer(10);

#ifndef NDEBUG
    DebugThreadID = std::this_thread::get_id();
#endif
}

TempestWidget::~TempestWidget()
{
}

QSize TempestWidget::sizeHint() const
{
    return QSize(200, 200);
}

void TempestWidget::attach(PreferredBackend* backend)
{
    m_Backend = backend;
    m_Backend->attach(GetWindowSystem(), m_Window);
}

void TempestWidget::paintEvent(QPaintEvent* event)
{
    if(m_Backend)
    {
        m_Backend->attach(GetWindowSystem(), m_Window);
    }
#ifndef NDEBUG
    TGE_ASSERT(DebugThreadID == std::this_thread::get_id(), "Should not be called from different thread");
#endif
    m_Backend->clearColorBuffer(0, Tempest::Vector4(1.0f, 0.0f, 1.0f, 1.0f));
    emit rendering();
    m_Window.swapBuffers();
}

void TempestWidget::timerEvent(QTimerEvent* event)
{
    update();
}
}