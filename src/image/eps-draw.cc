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

#include "tempest/image/eps-draw.hh"
#include "tempest/graphics/texture.hh"
#include "tempest/utils/system.hh"
#include "tempest/math/shapes.hh"

#include <algorithm>

namespace Tempest
{
EPSDraw::EPSDraw(const EPSImageInfo& info)
{
    m_Stream << "%!PS-Adobe-3.0 EPSF-3.0\n"
                "%%Creator: " << info.Creator << "\n"
                "%%BoundingBox: 0 0 " << info.Width << " " << info.Height << "\n"
                "%%LanguageLevel: 2\n"
                "%%Pages: 1\n"
                "%%DocumentData: Clean7Bit\n"
                "%%EndComments\n"
                "%%Page: 1 1\n"
                "save\n";
}

void EPSDraw::drawImage(const Texture& texture, float translate_x, float translate_y, float scale_x, float scale_y)
{
    auto& hdr = texture.getHeader();

    const size_t orig_data_size = Tempest::DataFormatElementSize(hdr.Format);
    const size_t orig_channel_count = Tempest::DataFormatChannels(hdr.Format);
    const size_t mod_channel_count = orig_channel_count != 4 ? orig_channel_count : 3; 
    const size_t elem_size = orig_data_size/orig_channel_count;
    const size_t mod_data_size = mod_channel_count*elem_size;
    
    m_Stream << "save\n";

    if(translate_x && translate_y)
    {
        m_Stream << translate_x << " " << translate_y << " translate\n";
    }

    m_Stream << (scale_x ? scale_x : hdr.Width) << " " << (scale_y ? scale_y : hdr.Height) << " scale\n"
             << hdr.Width << " " << hdr.Height << " " << elem_size*8
             << " [" << hdr.Width << " 0 0 " << hdr.Height << " 0 0]\n"
                "{<\n";

    auto* data = reinterpret_cast<const uint8_t*>(texture.getData());

    size_t line_chars = 0;
    for(size_t idx = 0, idx_end = hdr.Width*hdr.Height; idx < idx_end; ++idx)
    {
        auto elem_data = data + idx*orig_data_size;

        if((line_chars + mod_data_size*2) > 255)
        {
            m_Stream << "\n";
            line_chars = 0;
        }
        
        // Cope with endianness
        for(auto data_iter = elem_data, data_iter_end = elem_data + mod_channel_count*elem_size; data_iter < data_iter_end; ++data_iter)
        {
            m_Stream << std::hex << std::setw(2) << std::setfill('0') << (int)*data_iter;
            line_chars += 2;    
        }
    }

    m_Stream << "\n>}\n" << std::dec << std::setw(0)
             << "false " << mod_channel_count << " colorimage\n"
                "restore\n";
}

void EPSDraw::drawRect(const Rect2& rect, uint32_t fill_color, float line_thickness, uint32_t stroke_color)
{
    Vector2 axis0;
    FastSinCos(rect.Orientation, &axis0.y, &axis0.x);
    Vector2 axis1{ -axis0.y, axis0.x };

    auto p0 = rect.Center - rect.Size.x*axis0 - rect.Size.y*axis1;
    auto p1 = rect.Center + rect.Size.x*axis0 - rect.Size.y*axis1;
    auto p2 = rect.Center + rect.Size.x*axis0 + rect.Size.y*axis1;
    auto p3 = rect.Center - rect.Size.x*axis0 + rect.Size.y*axis1;

    m_Stream << "newpath\n"
             << p0.x << " " << p0.y << " moveto\n"
             << p1.x << " " << p1.y << " lineto\n"
             << p2.x << " " << p2.y << " lineto\n"
             << p3.x << " " << p3.y << " lineto\n"
                "closepath\n"
                "gsave\n"
             << rgbaR(fill_color)/255.0f << " " << rgbaG(fill_color)/255.0f << " " << rgbaB(fill_color)/255.0f << " setrgbcolor\n"
                "fill\n"
                "grestore\n"
             << line_thickness << " setlinewidth\n"
             << rgbaR(stroke_color)/255.0f << " " << rgbaG(stroke_color)/255.0f << " " << rgbaB(stroke_color)/255.0f << " setrgbcolor\n"
                "stroke\n";

}

void EPSDraw::drawPolygon(const Vector2* points, size_t data_size, uint32_t fill_color, float line_thickness, uint32_t stroke_color)
{
    if(data_size == 0)
        return;

    m_Stream << "newpath\n"
             << points->x << " " << points->y << " moveto\n";
    for(size_t i = 1; i < data_size; ++i)
    {
        auto& point = points[i];
        m_Stream << point.x << " " << point.y << " lineto\n";
    }

    m_Stream << "closepath\n"
                "gsave\n"
             << rgbaR(fill_color)/255.0f << " " << rgbaG(fill_color)/255.0f << " " << rgbaB(fill_color)/255.0f << " setrgbcolor\n"
             << "fill\n"
                "grestore\n"
             << line_thickness << " setlinewidth\n"
             << rgbaR(stroke_color)/255.0f << " " << rgbaG(stroke_color)/255.0f << " " << rgbaB(stroke_color)/255.0f << " setrgbcolor\n"
                "stroke\n";
}

void EPSDraw::drawLine(float x, float y, float xend, float yend, float line_thickness, uint32_t color)
{
    m_Stream << "newpath\n"
             << x << " " << y << " moveto\n"
             << xend << " " << yend << " lineto\n"
             << line_thickness << " setlinewidth\n"
             << rgbaR(color)/255.0f << " " << rgbaG(color)/255.0f << " " << rgbaB(color)/255.0f << " setrgbcolor\n"
                "stroke\n";
}

void EPSDraw::drawArc(float x, float y, float radius, float start_angle, float end_angle, float line_thickness, uint32_t color)
{
    m_Stream << "newpath\n"
             << line_thickness << " setlinewidth\n"
             << rgbaR(color)/255.0f << " " << rgbaG(color)/255.0f << " " << rgbaB(color)/255.0f << " setrgbcolor\n"
             << x << " " << y << " " << radius << " " << start_angle << " " << end_angle << " arc closepath\n"
               "stroke\n";
}

void EPSDraw::drawCircle(float x, float y, float radius, float line_thickness, uint32_t color)
{
    m_Stream << "newpath\n"
             << line_thickness << " setlinewidth\n"
             << rgbaR(color)/255.0f << " " << rgbaG(color)/255.0f << " " << rgbaB(color)/255.0f << " setrgbcolor\n"
             << x << " " << y << " " << radius << " 0 360 arc closepath\n"
               "stroke\n";
}

void EPSDraw::drawPath(const Vector2* points, size_t data_size, bool closepath, float line_thickness, uint32_t color)
{
    if(data_size == 0)
        return;

    m_Stream << "newpath\n"
             << points->x << " " << points->y << " moveto\n";
    for(size_t i = 1; i < data_size; ++i)
    {
        auto& point = points[i];
        m_Stream << point.x << " " << point.y << " lineto\n";
    }

    if(closepath)
    {
        m_Stream << "closepath\n";
    }

    m_Stream << line_thickness << " setlinewidth\n"
             << rgbaR(color)/255.0f << " " << rgbaG(color)/255.0f << " " << rgbaB(color)/255.0f << " setrgbcolor\n"
                "stroke\n";
}

void EPSDraw::drawText(const char* str, int fontsize, float translate_x, float translate_y, uint32_t color)
{
    m_Stream << "/Times-Roman findfont\n"
             << fontsize << " scalefont\n"
                "setfont\n"
                "newpath\n"
             << translate_x << " " << translate_y << " moveto\n"
             << rgbaR(color)/255.0f << " " << rgbaG(color)/255.0f << " " << rgbaB(color)/255.0f << " setrgbcolor\n"
             << "(" << str << ") show\n";            
}

bool EPSDraw::saveImage(const char* filename)
{
    {
    std::fstream fs(filename, std::ios::out|std::ios::trunc);
    if(!fs)
        return false;

    std::copy(std::istreambuf_iterator<char>(m_Stream),
              std::istreambuf_iterator<char>(),
              std::ostreambuf_iterator<char>(fs));

    fs << "restore\n"
          "showpage\n"
          "%%EOF\n";
    }

    System::Touch(filename);

    return true;
}
}