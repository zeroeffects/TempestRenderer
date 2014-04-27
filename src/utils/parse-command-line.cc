/*   The MIT License
 *   
 *   Tempest Engine
 *   Copyright (c) 2013 Zdravko Velinov
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

#include "tempest/utils/parse-command-line.hh"
#include "tempest/utils/logging.hh"

#include <cstring>

namespace Tempest
{
CommandLineOptsParser::CommandLineOptsParser(string name, bool unassociated_allowed)
    :   m_Name(name),
        m_UnassociatedAllowed(unassociated_allowed)
{
}

CommandLineOptsParser::~CommandLineOptsParser()
{
}

void CommandLineOptsParser::createOption(char short_name, string full_name, string description, bool has_arg)
{
    m_Opts[full_name] = OptionDescription(short_name, description, has_arg);
}

bool CommandLineOptsParser::parse(int argc, char* argv[])
{
    // TOODO: Better error reporting.
    for(int i = 1; i < argc; ++i)
    {
        TGE_ASSERT(argv[i], "Expected non-null command line option");
        char* opt = argv[i];
        if(opt[0] == '-')
        {
            if(opt[1] == '-')
            {
                auto iter = m_Opts.find(opt + 2);
                if(iter == m_Opts.end())
                {
                    Log(LogLevel::Error, "Unknown command line option: " + string(opt));
                    return false;
                }
            }
            else if(isalnum(opt[1]))
            {
                char short_name = opt[1];
                if(opt[2] != '\0')
                {
                    Log(LogLevel::Error, "Short command line option must be made of a single character: " + string(opt));
                    return false;
                }
                auto iter = std::find_if(m_Opts.begin(), m_Opts.end(), [short_name](const OptionDescriptionMap::value_type& desc){ return short_name == desc.second.getShortName(); });
                if(iter == m_Opts.end())
                {
                    Log(LogLevel::Error, "Unknown command line option: " + string(opt));
                    return false;
                }
                if(!iter->second.getHasValue())
                    continue;
                if(++i == argc)
                {
                    Log(LogLevel::Error, "Expected value after option: " + string(opt));
                    return false;
                }
                iter->second.setValue(argv[i]);
            }
            else
            {
                Log(LogLevel::Error, "Invalid command line option: " + string(opt));
                return false;
            }
        }
        else if(m_UnassociatedAllowed)
        {
            m_Unassociated.push_back(opt);
        }
        else
        {
            Log(LogLevel::Error, "Unexpected command line argument: " + string(opt));
            return false;
        }
    }
}
    
void CommandLineOptsParser::printHelp(std::ostream& os)
{
    os << m_Name << " [OPTIONS] args\n"
       << "Usage:\n";
    for(auto i = m_Opts.begin(); i != m_Opts.end(); ++i)
    {
        auto short_name = i->second.getShortName();
        os << "\t";
        if(short_name != FullNameOnly)
            os << "-" << short_name << ", ";
        os << "--" << i->first << "\t\t" << i->second.getDescription() << "\n";
    }
    
}
}