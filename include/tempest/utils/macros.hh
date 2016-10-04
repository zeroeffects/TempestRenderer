/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2010-2014 Zdravko Velinov
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

#ifndef _TEMPEST_MACROS_HH_
#define _TEMPEST_MACROS_HH_

#define _TO_STRING(s) # s
#define TO_STRING(s) _TO_STRING(s)

#define _CONCAT_MACRO(lhs, rhs) lhs##rhs
#define CONCAT_MACRO(lhs, rhs) _CONCAT_MACRO(lhs, rhs)

#define TEMPEST_MAKE_FOURCC(x, y, z, w) \
    (((x & 0xFFU))       | \
    ((y & 0xFFU) << 8U)  | \
    ((z & 0xFFU) << 16U) | \
    ((w & 0xFFU) << 24U))

#define TEMPEST_MAKE_EIGHTCC(c0, c1, c2, c3, c4, c5, c6, c7, c8) \
    (((c0 & 0xFFU))       | \
    ((c1 & 0xFFU) << 8U)  | \
    ((c2 & 0xFFU) << 16U) | \
    ((c3 & 0xFFU) << 24U))| \
    ((c4 & 0xFFU) << 32U))| \
    ((c5 & 0xFFU) << 40U))| \
    ((c6 & 0xFFU) << 48U))| \
    ((c7 & 0xFFU) << 56U))

#ifdef _MSC_VER
#   define ALIGN_1 __declspec(align(1))
#else
#   define ALIGN_1 __attribute__((aligned(1)))
#endif

#ifndef NDEBUG
#   define TGE_DEBUG_EXEC_ONCE static bool CONCAT_MACRO(test, __LINE__) = true; if(CONCAT_MACRO(test, __LINE__) ? CONCAT_MACRO(test, __LINE__) = false, true : false)
#else
#   define TGE_DEBUG_EXEC_ONCE if(0)
#endif

#define TGE_FIXED_ARRAY_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

#endif // _TEMPEST_MACROS_HH_