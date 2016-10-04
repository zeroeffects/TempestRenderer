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

#ifndef _TEMPEST_EPS_DRAW_HH_
#define _TEMPEST_EPS_DRAW_HH_

#include <sstream>
#include <cstdint>
#include <string>

namespace Tempest
{
class Texture;

struct EPSImageInfo
{
    uint32_t    Width,
                Height;
    const char* Creator = "Tempest";
};

union Vector2;
struct Rect2;

class EPSDraw
{
    std::stringstream m_Stream;
public:
    EPSDraw(const EPSImageInfo& info);

    std::string get() const { return m_Stream.str(); }

    void drawImage(const Texture& texture, float translate_x = 0, float translate_y = 0, float scale_x = 0, float scale_y = 0);
    void drawArc(float x, float y, float radius, float start_angle, float end_angle, float line_thickness = 1.0f, uint32_t stroke_color = 0);
    void drawCircle(float x, float y, float radius, float line_thickness = 1.0f, uint32_t stroke_color = 0);
    void drawLine(float x, float y, float xend, float yend, float line_thickness, uint32_t color = 0);
    void drawPath(const Vector2* points, size_t data_size, bool closepath, float line_thickness, uint32_t color = 0);
    void drawRect(const Rect2& rect, uint32_t fill_color = 0xFFFFFF, float line_thickness = 1.0f, uint32_t stroke_color = 0);
    void drawPolygon(const Vector2* points, size_t data_size, uint32_t fill_color = 0xFFFFFF, float line_thickness = 1.0f, uint32_t stroke_color = 0);

    void drawText(const char* str, int fontsize, float translate_x, float translate_y, uint32_t color = 0);

    bool saveImage(const char* filename);
};
}

#endif // _TEMPEST_EPS_DRAW_HH_