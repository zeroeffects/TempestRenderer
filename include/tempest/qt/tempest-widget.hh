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

#ifndef _TEMPEST_WIDGET_HH_
#define _TEMPEST_WIDGET_HH_

#include <QWidget>
#include <QWindow>
#include <QPaintEngine>
#include <memory>

#include "tempest/graphics/os-window.hh"
#include "tempest/graphics/preferred-backend.hh"


namespace Tempest
{
OSWindowSystem& GetWindowSystem();

class TempestWidget: public QWidget
{
    Q_OBJECT
    Q_DISABLE_COPY(TempestWidget)

    bool                        m_Rendering = false;

    PreferredBackend*           m_Backend = nullptr;
    PreferredWindow             m_Window;
public:
    TempestWidget(QWidget* parent = nullptr);
    ~TempestWidget();

    const PreferredWindow& getTempestWindow() const { return m_Window; }
    PreferredWindow& getTempestWindow() { return m_Window; }

    void attach(PreferredBackend* backend);

    virtual QSize sizeHint() const;

signals:
    void rendering();

protected:
    virtual void timerEvent(QTimerEvent* evt) override;
    virtual void paintEvent(QPaintEvent* evt) override;
};
}

#endif // _TEMPEST_WIDGET_HH_