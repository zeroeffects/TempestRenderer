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

#ifndef _EXR_VIEW_HH_
#define _EXR_VIEW_HH_
#include <QWidget>
#include <memory>
#include "tempest/graphics/texture.hh"

#undef None
enum class ExtractMode
{
    None,   
    Tangent,
    Binormal,
    Normal,
    Log,
    Heat,
    LogHeat
};

enum class NormalizationMode
{
    Manual,
    ScaleToRange,
    FitToRange
};


class EXRView: public QWidget
{
    Q_OBJECT

    std::unique_ptr<Tempest::Texture> m_Texture;

    std::unique_ptr<uint32_t[]>       m_Buffer;

    QImage                            m_CurrentImage;

    float                             m_Scale = 1.0f,
                                      m_Offset = 0.0f;

    QRect                             m_TargetRect;
    float                             m_ScaleRatio = 1.0f;
    
    int                               m_PreviousX = 0,
                                      m_PreviousY = 0;

    ExtractMode                       m_ExtractMode = ExtractMode::None;
    NormalizationMode                 m_Normalization = NormalizationMode::Manual;

public:
    EXRView(QWidget* parent);

    void setScale(float scale) { m_Scale = scale; updateTexture(); }
    void setOffset(float offset) { m_Offset = offset; updateTexture(); }
    float getScale() const { return m_Scale; }
    float getOffset() const { return m_Offset; }

    void setExtractMode(ExtractMode mode)
    {
        m_ExtractMode = mode;
        if(m_Normalization != NormalizationMode::Manual)
            updateScale();
        updateTexture();
    }
    void setNormalization(NormalizationMode mode)
    {
        m_Normalization = mode;
        if(mode != NormalizationMode::Manual)
            updateScale();
        updateTexture();
    }

    bool open(const QString& name);

    uint32_t getChannelCount() const { return m_Texture ? Tempest::DataFormatChannels(m_Texture->getHeader().Format) : 0; }

signals:
    void hovered(int image_x, int image_y, float* orig_value, float* displayed_value, unsigned component_count);

protected:
    virtual void paintEvent(QPaintEvent* evt);
    virtual void mouseMoveEvent(QMouseEvent* evt);
    virtual void wheelEvent(QWheelEvent* evt);

    void updateTexture();
    void updateScale();
};

#endif // _EXR_VIEW_HH_
