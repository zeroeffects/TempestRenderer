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

#ifndef _TEMPEST_IO_COMMAND_BUFFER_HH_
#define _TEMPEST_IO_COMMAND_BUFFER_HH_

#include "tempest/utils/types.hh"
#include "tempest/graphics/rendering-definitions.hh"

#include <memory>

namespace Tempest
{
class GLStorage;
class GLTexture;
class GLBuffer;

enum class IOCommandMode: uint16
{
    CopyBuffer,
    CopyTexture,
    CopyStorageToBuffer,
    CopyStorageToTexture,
    CopyBufferToStorage,
    CopyTextureToStorage
};

union GLResourceUnion
{
    GLStorage* Storage;
    GLTexture* Texture;
    GLBuffer*  Buffer;
};

struct GLIOCommand
{
    GLResourceUnion Source;
    GLResourceUnion Destination;
    union
    {
        struct
        {
            uint16 X,
                   Y;
        } SourceCoordinate;
        uint32 SourceOffset = 0;
    };

    union
    {
        struct
        {
            uint16 X,
                   Y;
        } DestinationCoordinate;
        uint32 DestinationOffset = 0;
    };

    uint16          SourceSlice = 0,
                    SourceMip = 0,
                    DestinationSlice = 0,
                    DestinationMip = 0,
                    Width = 1,
                    Height = 1,
                    Depth = 1;
    IOCommandMode   CommandType;
};

class GLIOCommandBuffer
{
    uint32                         m_IOCurrentCommand = 0;
    uint32                         m_IOCommandCount;
    std::unique_ptr<GLIOCommand[]> m_IOCommands;
    GLuint                         m_FBO;
public:
    typedef GLIOCommand IOCommandType;

    GLIOCommandBuffer(const IOCommandBufferDescription& cmd_desc);
    ~GLIOCommandBuffer();

    void clear();

    bool enqueueCommand(GLIOCommand command)
    {
        if(m_IOCurrentCommand == m_IOCommandCount)
            return false;
        m_IOCommands[m_IOCurrentCommand++] = command;
        return true;
    }
    
    void _executeCommandBuffer();
};
}

#endif // _TEMPEST_IO_COMMAND_BUFFER_HH_