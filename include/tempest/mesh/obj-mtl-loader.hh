/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2009 2010 2011 2012 Zdravko Velinov
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

#ifndef _OBJ_MTL_LOADER_HH_
#define _OBJ_MTL_LOADER_HH_

#include "tempest/math/vector3.hh"

namespace Tempest
{
namespace ObjMtlLoader
{
enum class IlluminationModel
{
    // Basic
    Diffuse,
    DiffuseAndAmbient,
    SpecularDiffuseAmbient,
    
    // Heavy duty
    ReflectionRayTrace,
    TransparencyRayTrace,
    ReflectionFresnelRayTrace,
    TransparencyRefractionNoFresnelRayTrace,
    TransparencyRefractionFresnelRayTrace,
    
    // Rasterizer
    Reflection,
    Transparency,
    Shadowmapping
};

// That's basically the generic ad hoc model that is not physically based at all.
// It compensates the lack of global illumination with ambient term.
struct Material
{
    string            Name;
    Vector3           Emission = Vector3(0.0f, 0.0f, 0.0f);                 // Ke
    Vector3           AmbientReflectivity = Vector3(1.0f, 1.0f, 1.0f);      // Ka
    Vector3           DiffuseReflectivity = Vector3(1.0f, 1.0f, 1.0f);      // Kd
    Vector3           SpecularReflectivity = Vector3(0.0f, 0.0f, 0.0f);     // Ks
    Vector3           TransmissionFilter = Vector3(0.0f, 0.0f, 0.0f);       // Tf
    IlluminationModel IllumModel = IlluminationModel::Diffuse;              // illum
    float             Dissolve = 0.0f;                                      // d (alpha)
    float             SpecularExponent = 1.0f;                              // Ns
    float             ReflectionSharpness = 1.0f;                           // sharpness
    float             RefractionIndex = 1.0f;                               // Ni
    string            AmbientReflectivityMap;                               // map_Ka
    string            DiffuseReflectivityMap;                               // map_Kd
    string            SpecularReflectivityMap;                              // map_Ks
    string            SpecularExponentMap;                                  // map_Ns
    string            DissolveMap;                                          // map_d
};
}
}

#endif // _OBJ_MTL_LOADER_HH_