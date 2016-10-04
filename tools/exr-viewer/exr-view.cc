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

#include <QPainter>
#include <QMouseEvent>

#include "exr-view.hh"
#include "tempest/image/exr-image.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/math/quaternion.hh"

const float WheelSpeed = 1e-2f;

EXRView::EXRView(QWidget* parent)
    :   QWidget(parent)
{
    this->setMouseTracking(true);
}

static void ComputeTargetRect(int window_width, int window_height, int image_width, int image_height, int* target_width, int* target_height, float* ratio)
{
    float target_aspect = (float)window_width/window_height;
    float source_aspect = (float)image_width/image_height;

    if(target_aspect > source_aspect)
    {
        *target_height = window_height + 0.5f;
        *target_width = window_height*source_aspect + 0.5f;
        *ratio = (float)*target_height/image_height;
    }
    else
    {
        *target_width = window_width + 0.5f;
        *target_height = window_width/source_aspect + 0.5f;
        *ratio = (float)*target_width/image_width;
    }
}

bool EXRView::open(const QString& name)
{
    m_Texture = decltype(m_Texture)(Tempest::LoadEXRImage(Tempest::Path(name.toStdString())));

    if(!m_Texture)
    {
        return false;
    }

    auto& hdr = m_Texture->getHeader();
    m_Buffer = decltype(m_Buffer)(new uint32_t[hdr.Width*hdr.Height]);

    m_CurrentImage = QImage(reinterpret_cast<uchar*>(m_Buffer.get()), hdr.Width, hdr.Height, QImage::Format_RGBA8888);

    if(m_Normalization != NormalizationMode::Manual)
        updateScale();

    auto size = this->size();
    auto window_width = size.width();
    auto window_height = size.height();

    int target_width, target_height;
    ComputeTargetRect(window_width, window_height, hdr.Width, hdr.Height, &target_width, &target_height, &m_ScaleRatio);

    int x = 0, y = 0;
    m_TargetRect = QRect(x, y, target_width, target_height);

    updateTexture();

    repaint();

    return true;
}

void EXRView::updateScale()
{
    if(!m_Texture)
        return;

    auto& hdr = m_Texture->getHeader();
    float max_value = -std::numeric_limits<float>::max(), min_value = std::numeric_limits<float>::max();
    switch(Tempest::DataFormatChannels(hdr.Format))
    {
    case 1:
    {
        for(uint32_t y = 0; y < hdr.Height; ++y)
            for(uint32_t x = 0; x < hdr.Width; ++x)
            {
                max_value = Maxf(max_value, m_Texture->fetchRed(x, y));
            }
    } break;
    case 2:
    {
        for(uint32_t y = 0; y < hdr.Height; ++y)
            for(uint32_t x = 0; x < hdr.Width; ++x)
            {
                max_value = Maxf(max_value, Tempest::MaxValue(m_Texture->fetchRG(x, y)));
                min_value = Minf(min_value, Tempest::MinValue(m_Texture->fetchRG(x, y)));
            }
    } break;
    case 3:
    {
        for(uint32_t y = 0; y < hdr.Height; ++y)
            for(uint32_t x = 0; x < hdr.Width; ++x)
            {
                max_value = Maxf(max_value, Tempest::MaxValue(m_Texture->fetchRGB(x, y)));
                min_value = Minf(min_value, Tempest::MinValue(m_Texture->fetchRGB(x, y)));
            }
    } break;
    case 4:
    {
        for(uint32_t y = 0; y < hdr.Height; ++y)
            for(uint32_t x = 0; x < hdr.Width; ++x)
            {
                max_value = Maxf(max_value, Tempest::MaxValue(m_Texture->fetchRGBA(x, y)));
                min_value = Minf(min_value, Tempest::MinValue(m_Texture->fetchRGBA(x, y)));
            }
    } break;
    }

    switch(m_ExtractMode)
    {
    case ExtractMode::Log:
    case ExtractMode::LogHeat:
    {
        min_value = logf(min_value);
        max_value = logf(max_value);
    } break;
    default: break;
    }

    switch(m_Normalization)
    {
    case NormalizationMode::ScaleToRange:
    {
        m_Offset = 0.0f;
        m_Scale = 1.0f/max_value;
    } break;
    case NormalizationMode::FitToRange:
    {
        m_Offset = -min_value;
        m_Scale = 1.0f/(max_value - min_value);
    } break;
    default: break;
    }
}

void EXRView::updateTexture()
{
    if(!m_Texture)
        return;

    auto& hdr = m_Texture->getHeader();
    switch(hdr.Format)
    {
    case Tempest::DataFormat::RGBA32F:
    {
        switch(m_ExtractMode)
        {
        default:
        case ExtractMode::None:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgba = m_Texture->fetchRGBA(x, y)*m_Scale + m_Offset;
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(rgba);
                }
        } break;
        case ExtractMode::Tangent:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgba = m_Texture->fetchRGBA(x, y);
                    auto quat = reinterpret_cast<Tempest::Quaternion&>(rgba);
                    auto tan = Tempest::Normalize(Tempest::ToTangent(quat));
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(tan*0.5f + 0.5f);
                }
        } break;
        case ExtractMode::Binormal:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgba = m_Texture->fetchRGBA(x, y);
                    auto quat = reinterpret_cast<Tempest::Quaternion&>(rgba);
                    auto binorm = Tempest::Normalize(Tempest::ToBinormal(quat));
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(binorm*0.5f + 0.5f);
                }
        } break;
        case ExtractMode::Normal:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgba = m_Texture->fetchRGBA(x, y);
                    auto quat = reinterpret_cast<Tempest::Quaternion&>(rgba);
                    auto norm = Tempest::Normalize(Tempest::ToNormal(quat));
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(norm*0.5f + 0.5f);
                }
        } break;
        case ExtractMode::Log:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgb = Tempest::Vector3Log(Tempest::ToVector3(m_Texture->fetchRGBA(x, y)));
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(rgb*0.5f + 0.5f);
                }
        } break;
        }
    } break;
    case Tempest::DataFormat::RGB32F:
    {
        switch(m_ExtractMode)
        {
        default:
        case ExtractMode::None:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgb = m_Texture->fetchRGB(x, y)*m_Scale + m_Offset;
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(rgb);
                }
        } break;
        case ExtractMode::Log:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rgb = Tempest::Vector3Log(m_Texture->fetchRGB(x, y))*m_Scale + m_Offset;
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(rgb);
                }
        } break;
        }
    } break;
    case Tempest::DataFormat::RG32F:
    {
        switch (m_ExtractMode)
        {
        default:
        case ExtractMode::None:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rg = m_Texture->fetchRG(x, y)*m_Scale + m_Offset;
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(Tempest::Vector3{ rg.x, rg.y, 0.0f });
                }
        } break;
        case ExtractMode::Log:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto rg = Tempest::Vector2Log(m_Texture->fetchRG(x, y))*m_Scale + m_Offset;
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = Tempest::ToColor(Tempest::Vector3{ rg.x, rg.y, 0.0f });
                }
        } break;
        }
    } break;
    case Tempest::DataFormat::R32F:
    {
        switch (m_ExtractMode)
        {
        default:
        case ExtractMode::None:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto r = m_Texture->fetchRed(x, y)*m_Scale + m_Offset;
                    auto color = Tempest::ToColor(Tempest::Vector3{ r, r, r });
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = color;
                }
        } break;
        case ExtractMode::Log:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    float orig_data = m_Texture->fetchRed(x, y);
                    auto r = orig_data > 0.0f ? logf(orig_data)*m_Scale + m_Offset : 0.0f;
                    auto color = Tempest::ToColor(Tempest::Vector3{ r, r, r });
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = color;
                }
        } break;
        case ExtractMode::Heat:
        {
             for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    auto r = m_Texture->fetchRed(x, y)*m_Scale + m_Offset;
                    auto color = Tempest::ToColor(Tempest::ColorCodeHSL4ToRGB(r));
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = color;
                }
        } break;
        case ExtractMode::LogHeat:
        {
            for(uint32_t y = 0; y < hdr.Height; ++y)
                for(uint32_t x = 0; x < hdr.Width; ++x)
                {
                    float orig_data = m_Texture->fetchRed(x, y);
                    auto r = orig_data > 0.0f ? logf(orig_data)*m_Scale + m_Offset : 0.0f;
                    auto color = Tempest::ToColor(Tempest::ColorCodeHSL4ToRGB(r));
                    m_Buffer[(hdr.Height - 1 - y)*hdr.Width + x] = color;
                }
        } break;
        }
    } break;
    }

    repaint();
}

void EXRView::wheelEvent(QWheelEvent* evt)
{
    if(!m_Texture)
        return;

    auto& hdr = m_Texture->getHeader();

    auto orig_target_width = hdr.Width*m_ScaleRatio,
         orig_target_height = hdr.Height*m_ScaleRatio;

    auto delta = evt->delta();
    if(delta < 0)
        m_ScaleRatio /= -WheelSpeed*delta;
    else
        m_ScaleRatio *= WheelSpeed*delta;

    auto target_width = hdr.Width*m_ScaleRatio,
         target_height = hdr.Height*m_ScaleRatio;

    auto pos = evt->pos();
    auto pos_x = pos.x(), pos_y = pos.y();

    auto size = this->size();
    auto window_width = size.width();
    auto window_height = size.height();
    if(pos_x < 0 || pos_y < 0 ||
       pos_x > window_width || pos_y > window_height)
    {
        m_TargetRect = QRect(m_TargetRect.x()*m_ScaleRatio, m_TargetRect.y()*m_ScaleRatio, target_width, target_height);
        return;
    }
    else
    {
        int new_x = pos_x - (pos_x - m_TargetRect.x())*target_width/orig_target_width,
            new_y = pos_y - (pos_y - m_TargetRect.y())*target_height/orig_target_height;

        m_TargetRect = QRect(new_x, new_y, target_width, target_height);
    }

    repaint();
}

void EXRView::paintEvent(QPaintEvent* event)
{
    QPainter painter(this);

    auto size = this->size();
    auto window_width = size.width();
    auto window_height = size.height();
    painter.fillRect(0, 0, window_width, size.height(), Qt::GlobalColor::black);

    if(!m_Texture)
    {
        return;
    }

    auto& hdr = m_Texture->getHeader();

    painter.drawImage(m_TargetRect, m_CurrentImage);
}

void EXRView::mouseMoveEvent(QMouseEvent* evt)
{
    if(!m_Texture)
        return;

    auto pos = evt->pos();
    auto pos_x = pos.x(), pos_y = pos.y();

    auto size = this->size();
    auto window_width = size.width();
    auto window_height = size.height();
    if(pos_x < 0 || pos_y < 0 ||
       pos_x > window_width || pos_y > window_height)
    {
        QWidget::mouseMoveEvent(evt);
        return;
    }

    if(evt->buttons() & Qt::RightButton)
    {
        m_TargetRect.translate((evt->x() - m_PreviousX), (evt->y() - m_PreviousY));

        repaint();
    }

    m_PreviousX = evt->x();
    m_PreviousY = evt->y();
    
    auto& hdr = m_Texture->getHeader();

    pos_x -= m_TargetRect.x();
    pos_y -= m_TargetRect.y();

    auto target_width = m_TargetRect.width(),
         target_height = m_TargetRect.height();

    if(pos_x >= target_width || pos_x < 0.0f || pos_y >= target_height || pos_y < 0.0f) 
    {
        emit hovered(-1, -1, nullptr, nullptr, 0);
        return;
    }

    Tempest::Vector2 tc { (float)pos_x*(hdr.Width - 1)/(target_width - 1), (float)pos_y*(hdr.Height - 1)/(target_height - 1) };

    uint32_t tci_x = Tempest::Clamp((uint32_t)tc.x, 0u, hdr.Width - 1u),
             tci_y = Tempest::Clamp((uint32_t)tc.y, 0u, hdr.Height - 1u);

    union CoordinateUnion
    {
        float               Red;
        Tempest::Vector2    RG;
        Tempest::Vector3    RGB;
        Tempest::Vector4    RGBA;
        float Components[4];
    };

    static_assert(sizeof(CoordinateUnion) == 4*sizeof(float), "Invalid union");

    CoordinateUnion orig_value;
    CoordinateUnion screen_value;

    uint32_t chan_count = 0;

    switch(hdr.Format)
    {
    case Tempest::DataFormat::RGBA32F:
    {
        switch(m_ExtractMode)
        {
        default:
        case ExtractMode::None:
        {
            orig_value.RGBA = m_Texture->fetchRGBA(tci_x, hdr.Height - 1 - tci_y);
            screen_value.RGBA = Tempest::ToVector4(m_Buffer[tci_y*hdr.Width + tci_x]);
            chan_count = 4;
        } break;
        case ExtractMode::Tangent:
        {
            auto rgba = m_Texture->fetchRGBA(tci_x, hdr.Height - 1 - tci_y);
            auto quat = reinterpret_cast<Tempest::Quaternion&>(rgba);
            auto tan = Tempest::Normalize(Tempest::ToTangent(quat));

            orig_value.RGB = tan;
            screen_value.RGB = Tempest::ToVector3(m_Buffer[tci_y*hdr.Width + tci_x]);
            chan_count = 3;
        } break;
        case ExtractMode::Binormal:
        {
            auto rgba = m_Texture->fetchRGBA(tci_x, hdr.Height - 1 - tci_y);
            auto quat = reinterpret_cast<Tempest::Quaternion&>(rgba);
            auto binorm = Tempest::Normalize(Tempest::ToBinormal(quat));

            orig_value.RGB = binorm;
            screen_value.RGB = Tempest::ToVector3(m_Buffer[tci_y*hdr.Width + tci_x]);
            chan_count = 3;
        } break;
        case ExtractMode::Normal:
        {
            auto rgba = m_Texture->fetchRGBA(tci_x, hdr.Height - 1 - tci_y);
            auto quat = reinterpret_cast<Tempest::Quaternion&>(rgba);
            auto norm = Tempest::Normalize(Tempest::ToNormal(quat));

            orig_value.RGB = norm;
            screen_value.RGB = Tempest::ToVector3(m_Buffer[tci_y*hdr.Width + tci_x]);
            chan_count = 3;
        } break;
        }
    } break;
    case Tempest::DataFormat::RGB32F:
    {
        orig_value.RGB = m_Texture->fetchRGB(tci_x, hdr.Height - 1 - tci_y);
        screen_value.RGB = Tempest::ToVector3(m_Buffer[tci_y*hdr.Width + tci_x]);
        chan_count = 3;
    } break;
    case Tempest::DataFormat::RG32F:
    {
        orig_value.RG = m_Texture->fetchRG(tci_x, hdr.Height - 1 - tci_y);
        const float scale = 1.0f/255.0f;
        auto color = m_Buffer[tci_y*hdr.Width + tci_x];
        screen_value.RG = { Tempest::rgbaR(color)*scale, Tempest::rgbaG(color)*scale };
        chan_count = 2;
    } break;
    case Tempest::DataFormat::R32F:
    {
        switch(m_ExtractMode)
        {
        default:
        {
            orig_value.Red = m_Texture->fetchRed(tci_x, hdr.Height - 1 - tci_y);
            const float scale = 1.0f/255.0f;
            auto color = m_Buffer[tci_y*hdr.Width + tci_x];
            screen_value.Red = Tempest::rgbaR(color)*scale;
            chan_count = 1;
        } break;
        case ExtractMode::Heat:
        {
            orig_value.Red = m_Texture->fetchRed(tci_x, hdr.Height - 1 - tci_y);
            screen_value.Red = orig_value.Red*m_Scale;
            chan_count = 1;
        } break;
        case ExtractMode::LogHeat:
        {
            orig_value.Red = m_Texture->fetchRed(tci_x, hdr.Height - 1 - tci_y);
            screen_value.Red = logf(orig_value.Red)*m_Scale;
            chan_count = 1;
        } break;
        }
    } break;
    }
    
    emit hovered(tci_x, hdr.Height - 1 - tci_y, orig_value.Components, screen_value.Components, chan_count);
}