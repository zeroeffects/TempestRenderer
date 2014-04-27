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

int TempestMain(int argc, char** argv);

#ifdef _WIN32
#include <cctype>
#include <cstdlib>
#include <cstdio>

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>

int CALLBACK WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    int argc = 0;
	char** argv = nullptr;
    char* end_ptr = lpCmdLine+strlen(lpCmdLine)+1;

    for(char* ptr = lpCmdLine, *arg_ptr; *ptr;)
    {
        if(isspace(*ptr))
        {
            ++ptr;
            continue;
        }
        argv = reinterpret_cast<char**>(realloc(argv, (argc+1)*sizeof(char*)));
        arg_ptr = argv[argc++] = reinterpret_cast<char*>(malloc(end_ptr-ptr));
        for(;!isspace(*ptr) && *ptr; ++ptr)
        {
            if(*ptr == '"')
                for(++ptr; *ptr != '"'; ++ptr)
                {
                    if(*ptr == '\0')
                    {
                        fprintf(stderr, "Error: expected \" at the end of input");
                        return EXIT_FAILURE;
                    }
                    else if(*ptr == '\\' && (ptr[1] == '"' || ptr[1] == '\\'))
                    {
                        ++ptr;
                        continue;
                    }

                    *arg_ptr = *ptr;
                    ++arg_ptr;
                }
            else if(*ptr == '\\' && (ptr[1] == '"' || ptr[1] == '\\'))
            {
                *arg_ptr = ptr[1];
                ++arg_ptr, ++ptr;
            }
            else
            {
                *arg_ptr = *ptr;
                ++arg_ptr;
            }
        }
        *arg_ptr = '\0';
    }

    TGE_INIT_TEMPEST_COMMON(argc, argv);
    
    TempestMain(argc, argv);

    while(argc--)
        free(argv[argc]);
    free(argv);

    return EXIT_SUCCESS;
}
#elif defined(LINUX)

#if !defined(NDEBUG) && defined(HAS_QT4)
#include <QApplication>
// Force Qt initialization because it may or may not get initialized depending
// on whether it is a debug build or release.
#   define TGE_INIT(argc, argv) QApplication qapp(argc, argv)
#else
#   define TGE_INIT(argc, argv)
#endif

int main(int argc, char* argv[])
{
    TGE_INIT(argc, argv);
    
    TempestMain(argc, argv);
}
#else
#   error "Unsupported platform"
#endif