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

#ifndef _TEMPEST_RENDERING_BACKED_HH_
#define _TEMPEST_RENDERING_BACKED_HH_

#include "tempest/math/vector4.hh"
#include "tempest/graphics/rendering-definitions.hh"

namespace Tempest
{
struct TextureDescription;
class ShaderProgram;
class InputLayout;
class Buffer;
class Texture;
class StateObject;
class RenderTarget;
class Framebuffer;
class ConstantBuffer;
class ShaderLoadingBackend;
class ConstantManager;
class DrawCommand;
class CommandBuffer;
 
struct RasterizerStates;
struct BlendStates;
struct DepthStencilStates;


struct FramebufferDescription
{
    RenderTargetBindPoint bindPoint;
    RenderTarget*         renderTarget;
};

/*! \brief The interface of the render device wrapper.
 *
 *  This interface closely maps to the structure of the modern graphics APIs. It might also contain
 *  some convenience objects to speed up development of the graphics pipeline draw call scheduling
 *  primitives. 
 * 
 *  This interface is not thread safe and it is suggested to use it from the thread that has created it.
 *  If a particular function is thread-safe, it is going to be explicitly stated; otherwise, you might
 *  randomly crash.
 */
class RenderingBackend
{
public:
    //! Constructor.
    explicit RenderingBackend()=default;

    //! Destructor.
    virtual ~RenderingBackend()=default;

    //! \remarks Copy constructor is not available.
    RenderingBackend(const RenderingBackend&)=delete;
    
    //! \remarks Assignment constructor.
    RenderingBackend& operator=(const RenderingBackend&)=delete;

    /*! \brief Create a render target (frame buffer) for off-screen rendering and later composition.
     * 
     *  Render targets represent a color, depth or stencil buffer used during the rendering of particular
     *  scene. They can be also bound as input textures to a particular shading stage.
     * 
     *  \param desc     contains description of the actual layout used for this render target.
     *  \param flags    some extraneous flags that are used only when creating render targets.
     *  \returns returns a new render target or nullptr on error. The error is automatically written to
     *           logging stream, if applicable.
     */
    virtual RenderTarget* createRenderTarget(const TextureDescription& desc, uint32 flags)=0;
    
    /*! \brief Create a framebuffer that can be assigned in a single call.
     * 
     *  Some APIs such as OpenGL implement render targets in this manner, so it might be beneficial to
     *  implement them in this manner because it reduces API calls. Also, Direct3D always sets render targets
     *  in this fashion, so it is highly beneficial to apply changes this way.
     * 
     *  The actual assignment of the object stages is done through its own interface.
     * 
     *  \returns returns a new render target batch.
     */    
    virtual Framebuffer* createFramebuffer()=0;
    
    /*! \brief Assign a framebuffer to the pipeline.
     * 
     *  This call makes the actual assignment. Make sure that you have assigned everything correctly to the render
     *  target batch before using it.
     * 
     *  \param rt_batch pointer the render target batch object.
     */
    virtual void setFramebuffer(Framebuffer* rt_batch)=0;
    
    /*! \brief Create a command buffer used for scheduling draw commands in a CPU friendly fashion.
     * 
     *  The command buffer might be safely accessed through other threads, which makes it convenient for building
     *  multi-threaded applications.
     * 
     *  \returns Returns a new command buffer object.
     */
    virtual CommandBuffer* createCommandBuffer()=0;
    
    /*! \brief Schedule a command buffer.
     *  
     *  This function schedules the command buffer that was hopefully built in another thread.
     * 
     *  \param cmd_buffer   a pointer to a command buffer object.
     */
    virtual void scheduleCommandBuffer(CommandBuffer* cmd_buffer)=0;
    
    /*! \brief Create a buffer object.
     * 
     *  Buffer objects are used as shader resources, command buffers and for output results of compute shaders.
     * 
     *  \param size     the size of the buffer in bytes.
     *  \param usage    hint how the buffer is going to be used to provide the appropriate optimizations.
     *  \param data     data used for automatic initialization (can be nullptr).
     *  \returns Returns a new buffer object.
     */
    virtual Buffer* createBuffer(size_t size, uint32 flags, const void* data)=0;
    
    /*! \brief Destroy a buffer object.
     * 
     *  Deallocates the buffer object. It might not get executed immediately because of stalling prevention.
     * 
     *  \param buffer   a pointer to buffer object.
     */
    virtual void destroyRenderResource(Buffer* buffer)=0;

    /*! \brief Create a shader loading backend.
     * 
     *  \returns Returns a pointer to the existing shader loading backend.
     */
    virtual ShaderLoadingBackend* getShaderBackend()=0;
    
    /*! \brief Create a texture object.
     * 
     *  Texture object represent a tiled or linear image in video memory. Depending on the specified flags
     *  they might reside within CPU or GPU memory. On some integrated graphics adapters they might be the
     *  same. Have in mind that it means less bandwidth in these cases.
     * 
     *  \param desc     a description of the data that is going to be stored in the memory associated
     *                  with this texture object.
     *  \param flags    hint how the memory is going to be used.
     *  \param data     data used for automatic initialization (can be nullptr).
     *  
     */
    virtual Texture* createTexture(const TextureDescription& desc, uint32 flags, const void* data = nullptr)=0;
    
    /*! \brief Destroy a texture object.
     * 
     *  Deallocates the texture object. It might not get executed immediately because of stalling prevention.
     * 
     *  \param texture  a pointer to the texture object.
     */
    virtual void destroyRenderResource(Texture* texture)=0;
    
    /*! \brief Create a state object.
     * 
     *  This kind of object is used for automatic setup of the complete pipeline state. Any of these states can be nullptr.
     *  This means that some parameters can be left out as-is.
     * 
     *  \param rasterizer_states    a pointer to raster state description.
     *  \param blend_states         a pointer to blend state description.
     *  \param depth_stencil_states a pointer to depth stencil state description.
     */
    virtual StateObject* createStateObject(const RasterizerStates* rasterizer_states, const BlendStates* blend_states, const DepthStencilStates* depth_stencil_states)=0;
    
    /*! \brief Set up scissor test rectangle.
     *  
     *  \param x      the left corner of the rectangle.
     *  \param y      the top corner of the rectangle.
     *  \param width  the horizontal size of the rectangle.
     *  \param height the vertical size of the rectangle.
     */
	virtual void setScissorRect(uint32 x, uint32 y, uint32 width, uint32 height) = 0;
    
    /*! \brief Set up the drawing rectangle.
     * 
     *  \param x      the left corner of the rectangle.
     *  \param y      the top corner of the rectangle.
     *  \param width  the horizontal size of the rectangle.
     *  \param height the vertical size of the rectangle.
     */
	virtual void setViewportRect(uint32 x, uint32 y, uint32 width, uint32 height) = 0;
    
    /*! \brief Clear up the color buffer with the specified color.
     * 
     *  The function does automatic conversion to the proper color format.
     * 
     *  \param idx    the index of the color buffer (-1 for all buffers).
     *  \param color  the color that is going to fill the color buffer.
     */
    virtual void clearColorBuffer(uint32 idx, const Vector4& color)=0;
    
    /*! \brief Clear up the depth stencil buffer with the specified depth and stencil value.
     * 
     *  \param depth   the depth value in non-linear format.
     *  \param stencil the stencil value.
     */
    virtual void clearDepthStencilBuffer(float depth=1.0f, uint8 stencil=0)=0;
};
}

#endif // _TEMPEST_RENDERING_BACKED_HH_