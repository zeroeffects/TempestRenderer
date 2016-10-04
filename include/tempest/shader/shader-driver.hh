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

#ifndef _GL_EFFECT_FILE_HH_
#define _GL_EFFECT_FILE_HH_

#include "tempest/shader/shader-ast.hh"

namespace Tempest
{
namespace Shader
{
class ShaderParser;

class Driver: public AST::Driver
{
    friend class ShaderParser;

    typedef std::vector<size_t> StackPointers;
    typedef std::vector<const Shader::Option*> OptionStackType;


    OptionStackType         m_OptionStack;
    StackPointers           m_OptionStackPointers;
    AST::StackType          m_ShaderBuiltIns,
                            m_FSBuiltIns;
    StackPointers           m_StackPointers;
public:
    Driver(FileLoader* loader);
     ~Driver();

    const Type* find(const std::string& name)
    {
        AST::Node* id = findIdentifier(name);
        if(id->getNodeType() == TGE_EFFECT_TYPE)
            return id->extract<Type>();
        else if(id->getNodeType() == TGE_EFFECT_TYPEDEF)
            return id->extract<Typedef>()->getType();
        return nullptr;
    }

    bool isOptionEnabled(const Optional* _opt);

    void beginOptionBlock() { m_OptionStackPointers.push_back(m_OptionStack.size()); }
    void endOptionBlock() { m_OptionStackPointers.pop_back(); }
    void pushOptionOnStack(const Shader::Option* _opt) { m_OptionStack.push_back(_opt); }

    void beginBlock();
    void endBlock();

    void beginShader(ShaderType shader_type);
    void endShader();

    bool parseFile(const std::string& filename);
    bool parseString(const char* content, size_t size, const std::string& filename="");
    
    template<class T, class... TArgs>
    AST::NodeT<AST::Reference<Type>> createStackType(Location loc, TArgs&&... args)
    {
        return pushOnStack(Shader::CreateTypeNode<T>(std::forward<TArgs>(args)...)) ?
            CreateNodeTyped<AST::Reference<Type>>(loc, m_ObjectPool.back().extract<Type>()) :
            AST::NodeT<AST::Reference<Type>>();
    }
    
    template<class T, class... TArgs>
    AST::NodeT<AST::Reference<Type>> createInternalType(Location loc, TArgs&&... args)
    {
        m_ObjectPool.push_back(Shader::CreateTypeNode<T>(std::forward<TArgs>(args)...));
        return CreateNodeTyped<AST::Reference<Type>>(loc, m_ObjectPool.back().extract<Type>());
    }
    
private:
    Type* extractType(AST::StackType& _stack, size_t idx)
    {
        return m_ObjectPool[_stack[idx]].extract<Type>();
    }
    
    template<class T>
    T* extractNode(AST::StackType& _stack, size_t idx)
    {
        return m_ObjectPool[_stack[idx]].extract<T>();
    }
    
    template<class T, class... TArgs>
    size_t createBuiltInType(TArgs&&... args)
    {
        size_t obj_idx = m_ObjectPool.size();
        m_ObjectPool.push_back(CreateTypeNode<T>(std::forward<TArgs>(args)...));
        return obj_idx;
    }
    
    template<class T, class... TArgs>
    size_t createBuiltInNode(TArgs&&... args)
    {
        size_t obj_idx = m_ObjectPool.size();
        m_ObjectPool.push_back(CreateNode<T>(TGE_DEFAULT_LOCATION, std::forward<TArgs>(args)...));
        return obj_idx;
    }
    
    size_t createFunctionSet(FunctionSet** func_set, std::string name)
    {
        size_t obj_idx = m_ObjectPool.size();
        m_ObjectPool.push_back(CreateNode<FunctionSet>(TGE_DEFAULT_LOCATION, name));
        *func_set = m_ObjectPool.back().extract<FunctionSet>();
        return obj_idx;
    }

    bool isOptionEnabledRecursive(const AST::Node* sub);
};
}
}

#endif /* _GL_EFFECT_FILE_HH_ */
