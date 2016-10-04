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

#ifndef _SPECTRUM_HH_
#define _SPECTRUM_HH_

#include "tempest/math/vector3.hh"
#include "tempest/math/vector4.hh"
#include "tempest/math/matrix3.hh"

namespace Tempest
{
#define MIN_CIE_SPECTRUM_WAVE_LENGTH 360
#define MAX_CIE_SPECTRUM_WAVE_LENGTH 830
#define RGB_SPECTRUM_SAMPLES 32

#define CIE_SPECTRUM_SAMPLE_COUNT (MAX_CIE_SPECTRUM_WAVE_LENGTH - MIN_CIE_SPECTRUM_WAVE_LENGTH + 1)

extern const float CIE_SpectrumCPU[3*CIE_SPECTRUM_SAMPLE_COUNT];

#ifndef SPECTRUM_SAMPLES
#   define SPECTRUM_SAMPLES 3
#   define DEFAULT_SPECTRUM
#   define MIN_SAMPLED_SPECTRUM_WAVE_LENGTH 360
#   define MAX_SAMPLED_SPECTRUM_WAVE_LENGTH 800
#endif

#if SPECTRUM_SAMPLES == 3
typedef Vector3 Spectrum;
#else
struct Spectrum
{
    float Components[SPECTRUM_SAMPLES];
};

#ifndef Array
#	define Array(x) x.Components
#endif
#endif

#ifdef DEFAULT_SPECTRUM
#   if SPECTRUM_SAMPLES == 6
EXPORT_CUDA_CONSTANT Spectrum SpectrumWaveLengths = 
{
    404, // violet
    446, // blue
    500, // green
    578, // yellow
    592, // orange
    650, // red
};
#   elif SPECTRUM_SAMPLES == 3
EXPORT_CUDA_CONSTANT Spectrum SpectrumWaveLengths =
{
    700, // red
	520, // green
	440 // blue
};
#   elif SPECTRUM_SAMPLES == 21
EXPORT_CUDA_CONSTANT Spectrum SpectrumWaveLengths = {
                              380,
                              400,
                              420,
                              440,
                              460,
                              480,
                              500,
                              520,
                              540,
                              560,
                              580,
                              600,
                              620,
                              640,
                              660,
                              680,
                              700,
                              720,
                              740,
                              760,
                              780,
};
#   endif
#endif

#if SPECTRUM_SAMPLES != 3
// There is code for validating this in spectrum.cc
EXPORT_CUDA_CONSTANT float CIE_YIntegral = 106.856834f;

// Refer to http://cvrl.ioo.ucl.ac.uk/database/data/cmfs/ciexyz31_1.txt
// You might not find it there, so you can find it in share/ also
extern Spectrum CIE_ReducedSpectrumCPU[3];

#ifdef __CUDA_ARCH__
#   define CIE_REDUCED_SPECTRUM CIE_ReducedSpectrumGPU
#   define SPECTRUM_CURVE_SET SpectrumCurveSetGPU
#else
#   define CIE_REDUCED_SPECTRUM CIE_ReducedSpectrumCPU
#   define SPECTRUM_CURVE_SET SpectrumCurveSetCPU
#endif

#endif

void InitSpectrum(); 

// This wrapping is to remind you that it is unlike the linear RGB space
struct CIEXYZ
{
	Vector3 XYZ;
};

// Refer to http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
// Reference white D65

/*
inline EXPORT_CUDA float ConvertLinearToSLuminance(float lum)
{
	return powf(lum, 1.0f/2.23333f);
}

inline EXPORT_CUDA float ConvertSLuminanceToLinear(float lum)
{
    return powf(lum, 2.23333f);
}
/*/
inline EXPORT_CUDA float ConvertLinearToSLuminance(float lum)
{
	if (lum <= 0.0031308f)
        return lum * 12.92f;
    else
        return 1.055f * powf(lum, 1.0f / 2.4f) - 0.055f;
}

inline EXPORT_CUDA float ConvertSLuminanceToLinear(float lum)
{
    if (lum <= 0.04045f)
        return lum / 12.92f;
    else
        return powf((lum + 0.055f) / 1.055f, 2.4f);
}
//*/

inline EXPORT_CUDA Vector3 ConvertLinearToSRGB(const Vector3& color)
{
	return Vector3{ ConvertLinearToSLuminance(color.x), ConvertLinearToSLuminance(color.y), ConvertLinearToSLuminance(color.z) };
}

inline EXPORT_CUDA Vector4 ConvertLinearToSRGB(const Vector4& color)
{
    return Vector4{ ConvertLinearToSLuminance(color.x), ConvertLinearToSLuminance(color.y), ConvertLinearToSLuminance(color.z), color.w };
}

inline EXPORT_CUDA Vector3 ConvertSRGBToLinear(const Vector3& color)
{
    return Vector3{ ConvertSLuminanceToLinear(color.x), ConvertSLuminanceToLinear(color.y), ConvertSLuminanceToLinear(color.z) };
}

inline EXPORT_CUDA Vector3 XYZToRGB(const CIEXYZ& color)
{
    const Matrix3 conv(Vector3{ 3.2404542f, -1.5371385f, -0.4985314f},
                       Vector3{-0.9692660f,  1.8760108f,  0.0415560f},
                       Vector3{ 0.0556434f, -0.2040259f,  1.0572252f});

    auto rgb_color = conv.transformRotationInverse(color.XYZ);

    return Vector3Max(rgb_color, 0.0f);
}

inline EXPORT_CUDA CIEXYZ RGBToXYZ(const Vector3& color)
{
    const Matrix3 conv(Vector3{0.4124564f, 0.3575761f, 0.1804375f},
                       Vector3{0.2126729f, 0.7151522f, 0.0721750f},
                       Vector3{0.0193339f, 0.1191920f, 0.9503041f});

	return CIEXYZ{conv.transformRotationInverse(color)};
}

inline EXPORT_CUDA Vector3 XYZToSRGB(const CIEXYZ& color)
{
    const Matrix3 conv(Vector3{ 3.2404542f, -1.5371385f, -0.4985314f},
                       Vector3{-0.9692660f,  1.8760108f,  0.0415560f},
                       Vector3{ 0.0556434f, -0.2040259f,  1.0572252f});

    Vector3 rgb_color = Vector3Max(conv.transformRotationInverse(color.XYZ), 0.0f);

    return ConvertLinearToSRGB(Vector3Max(rgb_color, 0.0f));
}

inline EXPORT_CUDA CIEXYZ SRGBToXYZ(const Vector3& color)
{
    const Matrix3 conv(Vector3{0.4124564f, 0.3575761f, 0.1804375f},
                       Vector3{0.2126729f, 0.7151522f, 0.0721750f},
                       Vector3{0.0193339f, 0.1191920f, 0.9503041f});

    Vector3 xyz_color = Vector3Max(conv.transformRotationInverse(ConvertSRGBToLinear(color)), 0.0f);

	return CIEXYZ{xyz_color};
}

struct HSL
{
    Vector3 Color;
};


inline EXPORT_CUDA Vector3 _impl_RGBToHSL(const Vector3& color)
{
    float cmin = MinValue(color),
          cmax = MaxValue(color),
          delta = cmax - cmin;
    float h, s, l;

    l = (cmax + cmin)*0.5f;
    if(delta == 0.0f)
    {
        h = 0.0f;
        s = 0.0f;
    }
    else
    {
        if(cmax == color.x)
        {
            h = MathPi*(fmodf((color.y - color.z)/delta, 6.0f))/3.0f;
        }
        else if(cmax == color.y)
        {
            h = MathPi*((color.z - color.x)/delta + 2.0f)/3.0f;
        }
        else
        {
            h = MathPi*((color.x - color.y)/delta + 4.0f)/3.0f;
        }
        s = delta / (1.0f - fabsf(2.0f*l - 1.0f));
    }

    if(h < 0.0f)
        h += 2.0f*MathPi;

    return { h, s, l };
}

inline EXPORT_CUDA HSL RGBToHSL(const Vector3& color)
{
    return { _impl_RGBToHSL(color) };
}

inline EXPORT_CUDA Vector3 _impl_HSLToRGB(const Vector3& color)
{
    float h = color.x,
          s = color.y,
          l = color.z;

    float c = (1.0f - fabsf(2.0f*l - 1.0f))*s;
    float x = c * (1.0f - fabsf(fmodf(h/(MathPi/3.0f), 2.0f) - 1.0f));
    float m = l - c*0.5f;

    if(h > 5.0f*2.0f*MathPi/6.0f)
    {
        return Vector3{ c + m, m, x + m };
    }
    else if(h > 4.0f*2.0f*MathPi/6.0f)
    {
        return Vector3{ x + m, m, c + m };
    }
    else if(h > 3.0f*2.0f*MathPi/6.0f)
    {
        return Vector3{ m, x + m, c + m };
    }
    else if(h > 2.0f*2.0f*MathPi/6.0f)
    {
        return Vector3{ m, c + m, x + m };
    }
    else if(h > 2.0f*MathPi/6.0f)
    {
        return Vector3{ x + m, c + m, m };
    }

    return Vector3{ c + m, x + m, m };
}

inline EXPORT_CUDA Vector3 HSLToRGB(const HSL& color)
{
    return _impl_HSLToRGB(color.Color);
}

struct YCbCr
{
    Vector3 Color;
};

struct YUV
{
    Vector3 Color;
};

inline EXPORT_CUDA YCbCr RGBToYCbCr(const Vector3& color)
{
    const Matrix3 conv(Vector3{ 0.183f,  0.614f,  0.062f},
                       Vector3{-0.101f, -0.339f,  0.439f},
                       Vector3{ 0.439f, -0.399f, -0.040f});

    const Vector3 off{ 16.0f/255.0f, 128.0f/255.0f, 128.0f/255.0f };

    return YCbCr{ Vector3Clamp(off + conv.transformRotationInverse(color), 0.0f, 1.0f) };
}

inline EXPORT_CUDA float RGBToLuminance(const Vector3& color)
{
    return 0.2126f*color.x + 0.7152f*color.y + 0.0722f*color.z;
}

inline EXPORT_CUDA Vector3 YCbCrToRGB(const YCbCr& color)
{
    const Matrix3 conv(Vector3{1.164f,    0.0f,  1.793f},
                       Vector3{1.164f, -0.213f, -0.533f},
                       Vector3{1.164f,  2.112f,    0.0f});

    const Vector3 off{ 16.0f/255.0f, 128.0f/255.0f, 128.0f/255.0f };

    return Vector3Clamp(conv.transformRotationInverse(color.Color - off), 0.0f, 1.0f);
}

inline EXPORT_CUDA Vector3 YUVToRGB(const YUV& color)
{
    const Matrix3 conv(Vector3{1.0f, 0.0f,       1.13983f},
                       Vector3{1.0f, -0.39465f, -0.5806f},
                       Vector3{1.0f, 2.03211f,   0.0f});
    
    return Vector3Clamp(conv.transformRotationInverse(color.Color), 0.0f, 1.0f);
}

inline EXPORT_CUDA Vector3 ColorCodeRGB6ToRGB(float value)
{
    const Vector3 color_table[6] =
    {
        { 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f },
        { 1.0f, 1.0f, 0.0f },
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 0.0f, 0.0f },
    };

    const unsigned max_elem = TGE_FIXED_ARRAY_SIZE(color_table) - 1; 
    float idx_f = value*max_elem;
    float fr = idx_f - FastFloor(idx_f);
    uint32_t lower = Clamp((uint32_t)idx_f, 0u, max_elem);
    uint32_t upper = Clamp((uint32_t)FastCeil(idx_f), 0u, max_elem);

    return Interpolate(color_table[lower], color_table[upper], fr);
}

inline EXPORT_CUDA Vector3 ColorCodeRGB4ToRGB(float value)
{
    const Vector3 color_table[4] =
    {
        { 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f },
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 0.0f, 0.0f },
    };

    const unsigned max_elem = TGE_FIXED_ARRAY_SIZE(color_table) - 1; 
    float idx_f = Clampf(value, 0.0f, 1.0f)*max_elem;
    float fr = idx_f - FastFloor(idx_f);
    uint32_t lower = Clamp((uint32_t)idx_f, 0u, max_elem);
    uint32_t upper = Clamp((uint32_t)FastCeil(idx_f), 0u, max_elem);

    return Interpolate(color_table[lower], color_table[upper], fr);
}

inline EXPORT_CUDA Vector3 ColorCodeHSL4ToRGB(float value)
{
    const Vector3 color_table[4] =
    {
        { MathPi*4.0f/3.0f, 1.0f, 0.5f },
        { MathPi/3.0f, 1.0f, 0.5f },
        { 0.0f, 1.0f, 0.5f },
        { 0.0f, 0.0f, 0.0f },
    };

    const unsigned max_elem = TGE_FIXED_ARRAY_SIZE(color_table) - 1; 
    float idx_f = Clampf(value, 0.0f, 1.0f)*max_elem;
    float fr = idx_f - FastFloor(idx_f);
    uint32_t lower = Clamp((uint32_t)idx_f, 0u, max_elem);
    uint32_t upper = Clamp((uint32_t)FastCeil(idx_f), 0u, max_elem);

    return _impl_HSLToRGB(Interpolate(color_table[lower], color_table[upper], fr));
}

#if SPECTRUM_SAMPLES == 3

#ifdef __CUDACC__
__device__ float* CIE_SpectrumGPU;
#endif

#ifdef __CUDA_ARCH__
#   define CIE_SPECTRUM CIE_SpectrumGPU
#else
#   define CIE_SPECTRUM CIE_SpectrumCPU
#endif

typedef Vector3 Spectrum;

inline EXPORT_CUDA Spectrum XYZToSpectrum(const CIEXYZ& color) { return XYZToRGB(color); }
inline EXPORT_CUDA CIEXYZ SpectrumToXYZ(const Vector3& spectrum) { return RGBToXYZ(spectrum); }

inline EXPORT_CUDA Vector3 SpectrumExp(const Vector3& spectrum)
{
    return Vector3Exp(spectrum);
}

inline EXPORT_CUDA Vector3 SpectrumCos(const Vector3& spectrum)
{
    return Vector3Cos(spectrum);
}

inline EXPORT_CUDA Vector3 SpectrumSinc(const Vector3& spectrum)
{
    return Vector3Sinc(spectrum);
}


inline EXPORT_CUDA Vector3 ToSpectrum(float val)
{
    return ToVector3(val);
}

#define SRGBToSpectrum ConvertSRGBToLinear

inline EXPORT_CUDA Spectrum RGBToSpectrum(const Vector3& color)
{
    return color;
}

inline EXPORT_CUDA Vector3 SpectrumToRGB(const Spectrum& spec)
{
    return spec;
}

inline EXPORT_CUDA Spectrum WavelengthToSpectrum(float wavelength)
{
    const int max_idx = MAX_CIE_SPECTRUM_WAVE_LENGTH - MIN_CIE_SPECTRUM_WAVE_LENGTH;
    if(wavelength < MIN_CIE_SPECTRUM_WAVE_LENGTH)
    {
        return XYZToRGB(CIEXYZ{ { CIE_SPECTRUM[0*CIE_SPECTRUM_SAMPLE_COUNT + 0], CIE_SPECTRUM[1*CIE_SPECTRUM_SAMPLE_COUNT + 0], CIE_SPECTRUM[2*CIE_SPECTRUM_SAMPLE_COUNT + 0] } });
    }
    else if(wavelength > MAX_CIE_SPECTRUM_WAVE_LENGTH)
    {
        return XYZToRGB(CIEXYZ{ { CIE_SPECTRUM[0*CIE_SPECTRUM_SAMPLE_COUNT + max_idx], CIE_SPECTRUM[1*CIE_SPECTRUM_SAMPLE_COUNT + max_idx], CIE_SPECTRUM[2*CIE_SPECTRUM_SAMPLE_COUNT + max_idx] } });
    }

    int wavelength_int = (int)wavelength;
    float mix_factor = wavelength - wavelength_int;

    int wavelength_off = wavelength_int - MIN_CIE_SPECTRUM_WAVE_LENGTH;
    int wavelength_off_upper = Mini(wavelength_off + 1, max_idx);

    Vector3 lower{ CIE_SPECTRUM[0*CIE_SPECTRUM_SAMPLE_COUNT + wavelength_off], CIE_SPECTRUM[1*CIE_SPECTRUM_SAMPLE_COUNT + wavelength_off], CIE_SPECTRUM[2*CIE_SPECTRUM_SAMPLE_COUNT + wavelength_off] };
    Vector3 upper{ CIE_SPECTRUM[0*CIE_SPECTRUM_SAMPLE_COUNT + wavelength_off_upper], CIE_SPECTRUM[1*CIE_SPECTRUM_SAMPLE_COUNT + wavelength_off_upper], CIE_SPECTRUM[2*CIE_SPECTRUM_SAMPLE_COUNT + wavelength_off_upper] };
    return XYZToRGB(CIEXYZ{ lower*(1.0f - mix_factor) + upper*mix_factor });
}
#else

enum SpectrumInCatalog
{
    WHITE_SPECTRUM,
    CYAN_SPECTRUM,
    MAGENTA_SPECTRUM,
    YELLOW_SPECTRUM,
    RED_SPECTRUM,
    GREEN_SPECTRUM,
    BLUE_SPECTRUM,
    SPECTRUM_COUNT
};

struct SpectrumCurve
{
    Spectrum Curve[SPECTRUM_COUNT];
    float Normalization;
};

extern SpectrumCurve SpectrumCurveSetCPU[2];

#ifdef __CUDACC__
__device__ Spectrum CIE_ReducedSpectrumGPU[3];
__device__ SpectrumCurve SpectrumCurveSetGPU[2];
#endif

inline EXPORT_CUDA Spectrum operator-(const Spectrum& spectrum)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = -spectrum.Components[i];
    return result;
}

inline EXPORT_CUDA Spectrum operator*(const Spectrum& lhs, const Spectrum& rhs)
{
    Spectrum result{};
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = lhs.Components[i] * rhs.Components[i];
    return result;
}

inline EXPORT_CUDA Spectrum& operator*=(Spectrum& lhs, const Spectrum& rhs)
{
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        lhs.Components[i] *= rhs.Components[i];
    return lhs;
}

inline EXPORT_CUDA Spectrum operator+(const Spectrum& lhs, const Spectrum& rhs)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = lhs.Components[i] + rhs.Components[i];
    return result;
}

inline EXPORT_CUDA Spectrum operator-(const Spectrum& lhs, const Spectrum& rhs)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = lhs.Components[i] - rhs.Components[i];
    return result;
}

inline EXPORT_CUDA Spectrum& operator+=(Spectrum& lhs, const Spectrum& rhs)
{
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        lhs.Components[i] += rhs.Components[i];
    return lhs;
}

inline EXPORT_CUDA Spectrum operator*(const Spectrum& spec, float scalar)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = spec.Components[i]*scalar;
    return result;
}

inline EXPORT_CUDA bool operator<(const Spectrum& spec, float scalar)
{
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        if(spec.Components[i] >= scalar)
            return false;
    return true;
}

inline EXPORT_CUDA bool operator>(const Spectrum& spec, float scalar)
{
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        if(spec.Components[i] <= scalar)
            return false;
    return true;
}
inline EXPORT_CUDA bool operator!=(const Spectrum& lhs, const Spectrum& rhs)
{
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        if(lhs.Components[i] == rhs.Components[i])
            return false;
    return true;
}


inline EXPORT_CUDA Spectrum& operator*=(Spectrum& spec, float scalar)
{
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        spec.Components[i] *= scalar;
    return spec;
}

inline EXPORT_CUDA Spectrum operator*(float scalar, const Spectrum& spec)
{
    return spec*scalar;
}

inline EXPORT_CUDA Spectrum operator/(const Spectrum& spec, float scalar)
{
    return spec*(1.0f/scalar);
}

inline EXPORT_CUDA Spectrum operator/(float scalar, const Spectrum& spec)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = scalar/spec.Components[i];
    return result;
}

inline EXPORT_CUDA Spectrum operator/(const Spectrum& lhs, const Spectrum& rhs)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = lhs.Components[i]/rhs.Components[i];
    return result;
}

inline EXPORT_CUDA Spectrum& operator/=(Spectrum& spec, float scalar)
{
    return spec *= (1.0f/scalar);
}

inline EXPORT_CUDA Spectrum ToSpectrum(float scalar)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = scalar;
    return result;
}

inline EXPORT_CUDA CIEXYZ SpectrumToXYZ(const Spectrum& spectrum)
{
    // TODO: SIMD
    Vector3 result{};
	float prev_wavelength = MIN_SAMPLED_SPECTRUM_WAVE_LENGTH;
	
	float dbg_total = 0.0f;

    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
    {
        float spectrum_value = spectrum.Components[i];
		float next_wavelength = i != SPECTRUM_SAMPLES - 1 ? (SpectrumWaveLengths.Components[i] + SpectrumWaveLengths.Components[i + 1])*0.5f : MAX_SAMPLED_SPECTRUM_WAVE_LENGTH;

		float delta_wavelength = next_wavelength - prev_wavelength;
		float scale = delta_wavelength/CIE_YIntegral;

        result.x += CIE_REDUCED_SPECTRUM[0].Components[i]*spectrum_value*scale;
        result.y += CIE_REDUCED_SPECTRUM[1].Components[i]*spectrum_value*scale;
        result.z += CIE_REDUCED_SPECTRUM[2].Components[i]*spectrum_value*scale;
		prev_wavelength = next_wavelength;

		dbg_total += delta_wavelength;
    }

	TGE_ASSERT(fabsf(dbg_total - (MAX_SAMPLED_SPECTRUM_WAVE_LENGTH - MIN_SAMPLED_SPECTRUM_WAVE_LENGTH)) < 1e-3f, "Invalid integration integration interval");

	return CIEXYZ{result};
}

inline EXPORT_CUDA float MaxValue(const Spectrum& spectrum)
{
    float max_value = spectrum.Components[0];
    for(size_t i = 1; i < SPECTRUM_SAMPLES; ++i)
    {
        float cur_value = spectrum.Components[i];
        if(max_value < cur_value)
            max_value = cur_value;
    }
    return max_value;
}

inline EXPORT_CUDA Spectrum Clamp(Spectrum& spectrum, float min_value, float max_value)
{
	Spectrum result;
	for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
		result.Components[i] = Clamp(spectrum.Components[i], min_value, max_value);
	return result;
}

inline EXPORT_CUDA Spectrum SpectrumMax(Spectrum& spectrum, float max_value)
{
    Spectrum result;
	for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
		result.Components[i] = Maxf(spectrum.Components[i], max_value);
	return result;
}

inline EXPORT_CUDA Spectrum SpectrumExp(const Spectrum& spectrum)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = expf(spectrum.Components[i]);
    return result;
}

inline EXPORT_CUDA Spectrum SpectrumCos(const Spectrum& spectrum)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = cosf(spectrum.Components[i]);
    return result;
}

inline EXPORT_CUDA Spectrum SpectrumSinc(const Spectrum& spectrum)
{
    Spectrum result;
    for(size_t i = 0; i < SPECTRUM_SAMPLES; ++i)
        result.Components[i] = Sinc(spectrum.Components[i]);
    return result;
}


inline EXPORT_CUDA bool ApproxEqual(const Spectrum& lhs, const Spectrum& rhs, float epsilon)
{
    for(uint32_t idx = 0; idx < SPECTRUM_SAMPLES; ++idx)
    {
        if(!Tempest::ApproxEqual(lhs.Components[idx], rhs.Components[idx], epsilon))
            return false;
    }
    return true;
}

inline EXPORT_CUDA Spectrum RGBToSpectrum(const Vector3& rgb, int light_source_D65 = 0)
{
    Spectrum result{};
    auto& spec = SPECTRUM_CURVE_SET[light_source_D65];
    if(rgb.x <= rgb.y && rgb.x <= rgb.z)
    {
        // Red is minimum
        result += rgb.x * spec.Curve[WHITE_SPECTRUM];
        if (rgb.y <= rgb.z)
        {
            result += (rgb.y - rgb.x) * spec.Curve[CYAN_SPECTRUM];
            result += (rgb.z - rgb.y) * spec.Curve[BLUE_SPECTRUM];
        }
        else
        {
            result += (rgb.z - rgb.x) * spec.Curve[CYAN_SPECTRUM];
            result += (rgb.y - rgb.z) * spec.Curve[GREEN_SPECTRUM];
        }
    }
    else if (rgb.y <= rgb.x && rgb.y <= rgb.z)
    {
        // Green is minimum
        result += rgb.y * spec.Curve[WHITE_SPECTRUM];
        if (rgb.x <= rgb.z)
        {
            result += (rgb.x - rgb.y) * spec.Curve[MAGENTA_SPECTRUM];
            result += (rgb.z - rgb.x) * spec.Curve[BLUE_SPECTRUM];
        }
        else
        {
            result += (rgb.z - rgb.y) * spec.Curve[MAGENTA_SPECTRUM];
            result += (rgb.x - rgb.z) * spec.Curve[RED_SPECTRUM];
        }
    }
    else
    {
        // Blue is minimum
        result += rgb.z * spec.Curve[WHITE_SPECTRUM];
        if (rgb.x <= rgb.y)
        {
            result += (rgb.x - rgb.z) * spec.Curve[YELLOW_SPECTRUM];
            result += (rgb.y - rgb.x) * spec.Curve[GREEN_SPECTRUM];
        }
        else
        {
            result += (rgb.y - rgb.z) * spec.Curve[YELLOW_SPECTRUM];
            result += (rgb.x - rgb.y) * spec.Curve[RED_SPECTRUM];
        }
    }
    result *= spec.Normalization;
    result = SpectrumMax(result, 0.0f);
	return result;
}

inline EXPORT_CUDA Spectrum SRGBToSpectrum(const Vector3& color, int light_source_D65 = 0)
{
    return RGBToSpectrum(ConvertSRGBToLinear(color), light_source_D65);
}

inline EXPORT_CUDA Vector3 SpectrumToRGB(const Spectrum& spec)
{
    return XYZToRGB(SpectrumToXYZ(spec));
}
#endif

inline EXPORT_CUDA uint32_t ToColor(const Vector3& vec)
{
    float x = Clamp(vec.x, 0.0f, 1.0f);
    float y = Clamp(vec.y, 0.0f, 1.0f);
    float z = Clamp(vec.z, 0.0f, 1.0f);
    return (uint32_t)(255.0f*x) | ((uint32_t)(255.0f*y) << 8u) | ((uint32_t)(255.0f*z) << 16u) | (255u << 24u);
}

inline EXPORT_CUDA uint32_t ToColor(const Vector4& vec)
{
    float x = Clamp(vec.x, 0.0f, 1.0f);
    float y = Clamp(vec.y, 0.0f, 1.0f);
    float z = Clamp(vec.z, 0.0f, 1.0f);
    float w = Clamp(vec.w, 0.0f, 1.0f);
    return (uint32_t)(255.0f*x) | ((uint32_t)(255.0f*y) << 8u) | ((uint32_t)(255.0f*z) << 16u) | ((uint32_t)(255.0f*w) << 24u);
}

inline CIEXYZ ColorSRGBToXYZ(uint32_t color)
{
    const float coef = (1.0f/255.0f);
    return SRGBToXYZ(Vector3{coef*rgbaR(color), coef*rgbaG(color), coef*rgbaB(color)});
}

inline EXPORT_CUDA Vector3 ToVector3(uint32_t color)
{
	const float scale = 1.0f/255.0f;
	return Vector3{rgbaR(color)*scale, rgbaG(color)*scale, rgbaB(color)*scale};
}

#if SPECTRUM_SAMPLES != 3
// TODO: Wavelength MAD?
inline EXPORT_CUDA Spectrum WavelengthToSpectrum(float wavelength)
{
    if(wavelength <= SpectrumWaveLengths.Components[0])
    {
        Spectrum spec{};
        spec.Components[0] = 1.0f;
        return spec;
    }
    else if(wavelength >= SpectrumWaveLengths.Components[SPECTRUM_SAMPLES - 1])
    {
        Spectrum spec{};
        spec.Components[SPECTRUM_SAMPLES - 1] = 1.0f;
        return spec;
    }

    uint32_t step = SPECTRUM_SAMPLES;
    uint32_t split = 0;

    do
    {
        step = (step + 1) >> 1;
        uint32_t new_split = split + step;

        if(new_split < SPECTRUM_SAMPLES && wavelength > Array(SpectrumWaveLengths)[new_split])
            split = new_split;
    } while(step > 1);

    float band_start_freq = Array(SpectrumWaveLengths)[split],
          band_end_freq = Array(SpectrumWaveLengths)[split + 1];

    float band = band_end_freq - band_start_freq;
    float ratio = (wavelength - band_start_freq)/band;

    Spectrum spec{};
    spec.Components[split] = 1.0f - ratio;
    spec.Components[split + 1] = ratio;

    return spec;
}
#endif
}

#endif // _SPECTRUM_HH_
