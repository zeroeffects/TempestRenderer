/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#include "tempest/qt/color-picker.hh"

#include <QColorDialog>
#include <QPalette>
#include <QPainter>

#include <algorithm>

ColorPicker::ColorPicker(QWidget* parent)
    :   QPushButton(parent)
{
    connect(this, SIGNAL(pressed()), this, SLOT(buttonPressed()));
}

void ColorPicker::buttonPressed()
{
    auto color = QColorDialog::getColor(m_Color, this, "Pick Color");
    if(!color.isValid())
        return;

    m_Color = color;

    update();
    emit colorChanged(m_Color);
}

void ColorPicker::paintEvent(QPaintEvent* evt)
{
    QPushButton::paintEvent(evt);

    const int border_size = 5;

    QSize size = this->size();
    QRect rect(border_size, border_size, std::max(0, size.width() - 2 * border_size), std::max(0, size.height() - 2 * border_size));

    QPainter painter(this);
    painter.fillRect(rect, m_Color);
}