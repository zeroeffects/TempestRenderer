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

#ifndef _PARSE_CMDLINE_HH_
#define _PARSE_CMDLINE_HH_

#include "tempest/utils/assert.hh"
#include "tempest/utils/file-system.hh"
#include "tempest/utils/system.hh"
#include "tempest/math/point2.hh"
#include "tempest/math/vector2.hh"
#include <limits>
#include <algorithm>
#include <unordered_map>

namespace Tempest
{
const char FullNameOnly = std::numeric_limits<char>::max();    

class OptionDescription
{
public:
    OptionDescription()
        :   m_ShortName(FullNameOnly),
            m_Arg(false) {}
    
    OptionDescription(char short_name, std::string description, bool has_arg, std::string default_value = "")
        :   m_ShortName(short_name),
            m_Description(description),
            m_Arg(has_arg),
            m_Value(default_value) {}
    
    OptionDescription(OptionDescription&& _desc)
        :   m_ShortName(_desc.m_ShortName),
            m_Description(std::move(_desc.m_Description)),
            m_Value(std::move(_desc.m_Value)),
            m_Arg(_desc.m_Arg) {}
    
    OptionDescription& operator=(OptionDescription&& _desc)
    {
        m_ShortName = _desc.m_ShortName;
        m_Description = std::move(_desc.m_Description);
        m_Value = std::move(_desc.m_Value);
        m_Arg = _desc.m_Arg;
        return *this;
    }
    
    OptionDescription(const OptionDescription& _desc)
        :   m_ShortName(_desc.m_ShortName),
            m_Description(_desc.m_Description),
            m_Value(_desc.m_Value),
            m_Arg(_desc.m_Arg) {}
    
    OptionDescription& operator=(const OptionDescription& _desc)
    {
        m_ShortName = _desc.m_ShortName;
        m_Description = _desc.m_Description;
        m_Value = _desc.m_Value;
        m_Arg = _desc.m_Arg;
        return *this;
    }
    
    template<class T>
    T convert() const
    {
        TGE_ASSERT(m_Arg, "The requested command line option does not support follow-up values");
        T val;
        std::stringstream ss;
        ss << m_Value;
        ss >> val;
        return val;
    }
    
    void setValue(std::string value) { m_Value = value; }
    std::string getValue() const { return m_Value; }
    bool isSet() const { return !m_Value.empty(); }

    char getShortName() const { return m_ShortName; }
    bool getHasValue() const { return m_Arg; }
    
    std::string getDescription() const { return m_Description; }
    
private:
    char   m_ShortName;
    std::string m_Description;
    std::string m_Value;
    bool   m_Arg;
};

class CommandLineOptsParser
{
    typedef std::unordered_map<std::string, OptionDescription> OptionDescriptionMap;
    typedef std::vector<std::string>                           UntypedValuesList;
    std::string                m_Name;
    bool                       m_UnassociatedAllowed;
    OptionDescriptionMap       m_Opts;
    UntypedValuesList          m_Unassociated;
public:
    CommandLineOptsParser(std::string name, bool unassociated_allowed);
     ~CommandLineOptsParser();
    
    void createOption(char short_name,std::string full_name, std::string description, bool has_arg, std::string default_value = "");
    
    template<typename T>
    T extract(const std::string& full_name) const
    {
        auto iter = m_Opts.find(full_name);
        TGE_ASSERT(iter != m_Opts.end(), "Unknown command line option");
        return iter->second.convert<T>();
    }

    bool isSet(const std::string& full_name)
    {
        auto iter = m_Opts.find(full_name);
        TGE_ASSERT(iter != m_Opts.end(), "Unknown command line option");
        return iter->second.isSet();
    }

    std::string extractString(const std::string& full_name) const
    {
        auto iter = m_Opts.find(full_name);
        TGE_ASSERT(iter != m_Opts.end(), "Invalid option");
        if(iter == m_Opts.end())
            return {};

        return iter->second.getValue();
    }
    
    std::string getUnassociatedArgument(size_t idx) { return m_Unassociated[idx]; }
    
    template<class T>
    T extractUnassociatedArgument(size_t idx)
    {
        T val;
        std::stringstream ss;
        ss << m_Unassociated[idx];
        ss >> val;
        return val;
    }
    
    size_t getUnassociatedCount() const { return m_Unassociated.size(); }
    
    bool parse(int argc, char* argv[]);
    
    void printHelp(std::ostream& os);
};

union Vector3;

bool ParseResolution(const std::string& display, uint32_t* out_width, uint32_t* out_height);
bool ParseDirection(const char* str, Vector3* out_vec);
bool ParseCommaSeparatedPoints(const char* str, std::vector<Point2>* out_vec);

template<class TVector>
bool ParseCommaSeparatedVectors(const char* str, std::vector<TVector>* out_vec);

template<class... TArgs>
void GenerateError(TArgs&&... args)
{
    auto exe_path = System::GetExecutablePath();
    Tempest::Log(LogLevel::Error, Path(exe_path).filename(), ": error: ", args...);
}
}

#endif // _PARSE_CMDLINE_HH_
