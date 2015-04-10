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

#ifndef TEMPEST_GL_RENDERING_BACKEND_HH_
#define TEMPEST_GL_RENDERING_BACKEND_HH_

#include "tempest/graphics/rendering-backend.hh"
#include "tempest/graphics/opengl-backend/gl-state-object.hh"
#include "tempest/graphics/os-window.hh"

#include <unordered_set>
#include <memory>

namespace Tempest
{
/*! \brief The interface of the render device wrapper.
 *
 *  It implements the backend for the OpenGL graphics library.
 * 
 *  This interface is not thread safe and it is suggested to use it from the thread that has created it.
 *  If a particular function is thread-safe, it is going to be explicitly stated; otherwise, you might
 *  randomly crash.
 */
class GLRenderTarget;
class GLFramebuffer;
class GLCommandBuffer;
class GLBuffer;
class GLStateObject;
class GLTexture;
class GLShaderProgram;
class GLInputLayout;
class GLWindow;
class GLStorage;
class GLFence;
class GLIOCommandBuffer;
class BakedResourceTable;

struct GLBlendStates;
struct GLRasterizerStates;
struct GLDepthStencilStates;

#ifdef LINUX
typedef std::shared_ptr<GLXFBConfig> GLXFBConfigPtr;
#endif

class GLRenderingBackend
{
    template<class T>
    struct PackDeleter
    {
        void operator()(T* ptr) { DestroyPackedData(ptr); }
    };

    template<class T, class TDeleter>
    struct CompareIndirect
    {
        bool operator()(const std::unique_ptr<T, TDeleter>& lhs, const std::unique_ptr<T, TDeleter>& rhs) const;
    };

    template<class T, class TDeleter>
    struct HashIndirect
    {
        size_t operator()(const std::unique_ptr<T, TDeleter>& state_obj) const;
    };

    template<class T, class TDeleter = std::default_delete<T>>
    using StateHashMap = std::unordered_set < std::unique_ptr<T, TDeleter>, HashIndirect<T, TDeleter>, CompareIndirect<T, TDeleter> >;

    typedef StateHashMap<GLStateObject> GLStateObjectMap;
    typedef StateHashMap<GLBlendStates> GLBlendStatesMap;
    typedef StateHashMap<GLRasterizerStates> GLRasterizerStatesMap;
    typedef StateHashMap<GLDepthStencilStates> GLDepthStencilStatesMap;
    typedef StateHashMap<GLInputLayout, PackDeleter<GLInputLayout>> GLInputLayoutMap;
    const GLStateObject*    m_CurrentStateObject = nullptr;
    GLStateObjectMap        m_StateObjects;
    GLBlendStatesMap        m_BlendStates;
    GLRasterizerStatesMap   m_RasterizerStates;
    GLDepthStencilStatesMap m_DepthStencilStates;
    GLInputLayoutMap        m_InputLayoutMap;

    GLBlendStates           m_DefaultBlendState;
    GLRasterizerStates      m_DefaultRasterizerStates;
    GLDepthStencilStates    m_DefaultDepthStencilStates;

#ifndef TEMPEST_DISABLE_TEXTURE_BINDLESS
    GLuint                  m_TexturesTable = 0;
    uint32                  m_ActiveTextures = 0;
#endif

#ifdef LINUX
    // Because there is not a cannonical description what it should contain we pessimize the code.
    GLXFBConfigPtr  m_FBConfig;
    GLXContext      m_GLXContext = GLXContext();
#elif defined(_WIN32)
    HDC             m_DC = nullptr;
    HGLRC           m_HGLRC = nullptr;
#endif
    OSWindowSystem* m_Display = nullptr;
public:
    typedef GLRenderTarget    RenderTargetType;
    typedef GLCommandBuffer   CommandBufferType;
    typedef GLFramebuffer     FramebufferType;
    typedef GLBuffer          BufferType;
    typedef GLStateObject     StateObjectType;
    typedef GLTexture         TextureType;
    typedef GLStorage         StorageType;
    typedef GLShaderProgram   ShaderProgramType;
    typedef GLIOCommandBuffer IOCommandBufferType;
    typedef GLsync            FenceType;
    
    //! Constructor.
    explicit GLRenderingBackend();

    //! Destructor.
     ~GLRenderingBackend();

    //! \remarks Copy constructor is not available.
    GLRenderingBackend(const GLRenderingBackend&)=delete;
    
    //! \remarks Assignment constructor.
    GLRenderingBackend& operator=(const GLRenderingBackend&)=delete;

    /*! \brief Attaches the rendering backend to a window.
     */
    bool attach(OSWindowSystem& wnd_sys, GLWindow& gl_ctx);

    /*! \brief Setups the initial state. Requires the library to be completely loaded.
     *  \remarks Load library functions before calling this function.
     */
    void init();

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
    GLRenderTarget* createRenderTarget(const TextureDescription& desc, uint32 flags);
    
    /*! \brief Create a batch of render targets that can be assigned in a single call.
     * 
     *  Some APIs such as OpenGL implement render targets in this manner, so it might be beneficial to
     *  implement them in this manner because it reduces API calls. Also, Direct3D always sets render targets
     *  in this fashion, so it is highly beneficial to apply changes this way.
     * 
     *  The actual assignment of the object stages is done through its own interface.
     * 
     *  \returns returns a new render target batch.
     */    
    GLFramebuffer* createFramebuffer();
    
    /*! \brief Assign a framebuffer to the pipeline.
     * 
     *  This call makes the actual assignment. Make sure that you have assigned everything correctly to the render
     *  target batch before using it.
     * 
     *  \param rt_batch pointer the render target batch object.
     */
    void setFramebuffer(GLFramebuffer* rt_batch);
    
    /*! \brief Create a command buffer used for scheduling draw commands in a CPU friendly fashion.
     * 
     *  The command buffer might be safely accessed through other threads, which makes it convenient for building
     *  multi-threaded applications.
     *  
     *  \param cmd_buf_desc     Used for specifying command buffer dimensions.
     *  \returns Returns a new command buffer object.
     */
    GLCommandBuffer* createCommandBuffer(const CommandBufferDescription& cmd_buf_desc);
    
    /*! \brief Submit a command buffer.
     *  
     *  This function schedules the command buffer that was can be built in another thread.
     * 
     *  \param cmd_buffer   a pointer to a command buffer object.
     */
    void submitCommandBuffer(GLCommandBuffer* cmd_buffer);
    
    /*! \brief Create an I/O command buffer used for scheduling asynchronous copy operations.
     */
    GLIOCommandBuffer* createIOCommandBuffer(const IOCommandBufferDescription& cmd_buf_desc);

    /*! \brief Destroy an I/O command buffer.
     */
    void destroyRenderResource(GLIOCommandBuffer* cmd_buffer);
    
    /*! \brief Submit an I/O command buffer.
    *
    *  This function schedules the I/O command buffer that can be built in another thread.
    *
    *  \param cmd_buffer   a pointer to a command buffer object.
    */
    void submitCommandBuffer(GLIOCommandBuffer* cmd_buffer);

    /*! \brief Destroy a command buffer object.
     * 
     *  Deallocates the command buffer object. It might not get executed immediately because of stalling prevention.
     * 
     *  \param buffer   a pointer to buffer object.
     */
    void destroyRenderResource(GLCommandBuffer* buffer);
    
    /*! \brief Create a buffer object.
     * 
     *  Buffer objects are used as shader resources, command buffers and for output results of compute shaders.
     * 
     *  \param type     the type of buffer.
     *  \param size     the size of the buffer in bytes.
     *  \param usage    hint how the buffer is going to be used to provide the appropriate optimizations.
     *  \param data     data used for automatic initialization (can be nullptr).
     *  \returns Returns a new buffer object.
     */
    GLBuffer* createBuffer(size_t size, ResourceBufferType buffer_type, uint32 flags = RESOURCE_STATIC_DRAW, const void* data = nullptr);
    
    /*! \brief Set textures in a single call.
     *
     *  The layout and the methods of setup might depend on the OpenGL capabilities. It might be done via bindless
     *  textures or old style bind points.
     *
     *  \param resource_table   packed texture handles.
     */
    void setTextures(const BakedResourceTable* resource_table);

    // TODO: Make signatures...
    void setConstantBuffer(size_t idx, const GLBuffer* buf);

    /*! \brief Set the number of active textures.
     *
     *  The active number of textures may be controlled, if relevant.
     *
     *  \param num_textures     the number of requested active textures.
     */
    void setActiveTextures(uint32 num_textures);

    /*! \brief Destroy a buffer object.
     * 
     *  Deallocates the buffer object. It might not get executed immediately because of stalling prevention.
     * 
     *  \param buffer   a pointer to buffer object.
     */
    void destroyRenderResource(GLBuffer* buffer);

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
    GLTexture* createTexture(const TextureDescription& desc, uint32 flags = RESOURCE_STATIC_DRAW, const void* data = nullptr);
    
    /*! \brief Destroy a texture object.
     * 
     *  Deallocates the texture object. It might not get executed immediately because of stalling prevention.
     * 
     *  \param texture  a pointer to the texture object.
     */
    void destroyRenderResource(GLTexture* texture);
    
    /*! \brief Create a state object.
     * 
     *  This kind of object is used for automatic setup of the complete pipeline state. Any of these states can be nullptr.
     *  This means that some parameters can be left out as-is.
     * 
     *  \param rasterizer_states    a pointer to raster state description.
     *  \param blend_states         a pointer to blend state description.
     *  \param depth_stencil_states a pointer to depth stencil state description.
     */
    GLStateObject* createStateObject(DataFormat*,
                                     size_t,
                                     DataFormat,
                                     GLShaderProgram* shader_program,
                                     DrawModes primitive_type = DrawModes::TriangleList,
                                     const RasterizerStates* rasterizer_states = nullptr,
                                     const BlendStates* blend_states = nullptr,
                                     const DepthStencilStates* depth_stencil_state = nullptr
                                     );
    
    /*! \brief Destroy a state object.
    *
    *   \remarks In this implementation it does nothing because it has elaborate duplication tracking mechanism which guarantees that every state
    *            is unique. It is expected that the overall unique states are going to be constant and/or small amount.
    *
    *   \param state_obj  a pointer to the state object.
    */
    void destroyRenderResource(GLStateObject* state_obj) {}

    /*! \brief Set up scissor test rectangle.
     *  
     *  \param x      the left corner of the rectangle.
     *  \param y      the top corner of the rectangle.
     *  \param width  the horizontal size of the rectangle.
     *  \param height the vertical size of the rectangle.
     */
	void setScissorRect(uint32 x, uint32 y, uint32 width, uint32 height);
    
    /*! \brief Set up the drawing rectangle.
     * 
     *  \param x      the left corner of the rectangle.
     *  \param y      the top corner of the rectangle.
     *  \param width  the horizontal size of the rectangle.
     *  \param height the vertical size of the rectangle.
     */
	void setViewportRect(uint32 x, uint32 y, uint32 width, uint32 height);
    
    /*! \brief Clear up the color buffer with the specified color.
     * 
     *  The function does automatic conversion to the proper color format.
     * 
     *  \param idx    the index of the color buffer (-1 for all buffers).
     *  \param color  the color that is going to fill the color buffer.
     */
    void clearColorBuffer(uint32 idx, const Vector4& color);
    
    /*! \brief Clear up the depth stencil buffer with the specified depth and stencil value.
     * 
     *  \param depth   the depth value in non-linear format.
     *  \param stencil the stencil value.
     */
    void clearDepthStencilBuffer(float depth=1.0f, uint8 stencil=0);

    /*! \brief Create a storage buffer.
     *  
     *  Storage buffers are allocated in CPU visible memory and they are used for transferring
     *  resource data to GPU visible memory in asynchronous fashion.
     *
     *  \param storage_type     the type of the storage buffer.
     *  \param size             the size of the storage buffer in bytes.
     */
    GLStorage* createStorageBuffer(StorageMode storage_type, uint32 size);

    /*! \brief Destroy a storage buffer.
     *
     *  \param storage           a pointer to the storage buffer.
     */
    void destroyRenderResource(GLStorage* storage);

    /*! \brief Create fence object used for synchronization between CPU and GPU.
     */
    FenceType* createFence();

    /*! \brief Push a fence object to command stream.
     */
    void pushFence(FenceType* fence);

    /*! \brief Destroy fence object
     */
    void destroyRenderResource(FenceType* fence);

    /*! \brief Wait on a fence object. 
     *  
     *  \remarks Fence objects can't be used outside of rendering thread for platform specific reasons.
     */
    void waitFence(FenceType* fence);

    /*! \brief Get texture handle size.
     *
     *  It is used for building baked texture tables by hand.
     */
    uint32 getTextureHandleSize();
};
}

#endif // TEMPEST_GL_RENDERING_BACKEND_HH_