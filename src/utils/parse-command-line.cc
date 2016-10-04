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
#include "tempest/math/vector3.hh"

#include <cstring>
#include <iomanip>

namespace Tempest
{
bool ParseResolution(const std::string& display, uint32_t* out_width, uint32_t* out_height)
{
    auto delim_pos = display.find('x');
    if(delim_pos == std::string::npos)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid display size specified. Use format NxM, e.g. 400x400.");
        return false;
    }

    uint32_t image_width, image_height;

    std::stringstream ss;
    auto sub_width = display.substr(0, delim_pos);
    ss << sub_width;
    ss >> image_width;
    if(!ss)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to parse the specified display resolution: ", display);
        return false;
    }

    ss.clear();
    auto sub_height = display.substr(delim_pos + 1);
    ss << sub_height;
    ss >> image_height;
    if(!ss)
    {
        Tempest::Log(Tempest::LogLevel::Error, "failed to parse the specified display resolution: ", display);
        return false;
    }

    if(image_width == 0 || image_height == 0)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid display resolution specified: ", display);
        return false;
    }

    *out_width = image_width;
    *out_height = image_height;

    return true;
}

bool ParseDirection(const char* str, Tempest::Vector3* out_vec)
{
    Tempest::Vector3 parsed_vec;
    char* end_of_str = nullptr;
	size_t parsed_comp_idx = 0;
    do
	{
	    end_of_str = nullptr;
		float parsed_float = strtof(str, &end_of_str);
        if(parsed_float == HUGE_VALF)
        {
            Tempest::Log(Tempest::LogLevel::Info, "failed to parse direction");
            return false;
        }

        parsed_vec.Components[parsed_comp_idx] = parsed_float;
		str = end_of_str + 1;
		++parsed_comp_idx;
    } while(parsed_comp_idx < TGE_FIXED_ARRAY_SIZE(parsed_vec.Components) && end_of_str && *end_of_str);

    if(*end_of_str)
    {
        Tempest::Log(Tempest::LogLevel::Error, "invalid vector declaration");
        return false;
    }
   
    *out_vec = parsed_vec;
    return true;
}

bool ParseCommaSeparatedPoints(const char* str, std::vector<Point2>* out_vec)
{
    uint32_t parsed_comp = 0;
    Point2 parsed_point{};
    bool parsing = false;
    std::vector<Point2> result_vec;
    for(char c; c = *str; ++str)
    {
        if(isdigit(c))
        {
            parsing = true;
            auto& cur_comp = parsed_point.Components[parsed_comp];
            cur_comp = cur_comp*10 + (uint32_t)(c - '0');
        }
        else if(c == ':')
        {
            ++parsed_comp;
            if(parsed_comp >= TGE_FIXED_ARRAY_SIZE(parsed_point.Components))
            {
                Tempest::Log(Tempest::LogLevel::Error, "too many components specified for point");
                return false;
            }
        }
        else if(c == ',')
        {
            result_vec.push_back(parsed_point);
            parsed_point = {};
            parsed_comp = 0;
            parsing = false;
        }
        else if(!isspace(c))
        {
            Tempest::Log(Tempest::LogLevel::Error, "invalid character in point declaration");
            return false;
        }
    }

    if(parsed_comp < TGE_FIXED_ARRAY_SIZE(parsed_point.Components) - 1 && parsing)
    {
        Tempest::Log(Tempest::LogLevel::Error, "incomplete number at point declaration");
        return false;
    }
    
    if(parsed_comp == TGE_FIXED_ARRAY_SIZE(parsed_point.Components) - 1)
    {
        result_vec.push_back(parsed_point);
    }

    *out_vec = std::move(result_vec);
    return true;
}

template<class TVector>
bool ParseCommaSeparatedVectors(const char* str, std::vector<TVector>* out_vec)
{
    uint32_t parsed_comp = 0;
    TVector parsed_point{};
    bool parsing = false;
    std::vector<TVector> result_vec;
    for(char c; c = *str; ++str)
    {
		parsed_point.Components[parsed_comp] = strtof(str, const_cast<char**>(&str));
        if(parsed_point.Components[parsed_comp] == HUGE_VALF)
        {
            Tempest::Log(Tempest::LogLevel::Info, "failed to parse direction");
            return false;
        }

        parsing = true;

        if(!*str)
            break;

        c = *str;

        if(c == ':')
        {
            ++parsed_comp;
            if(parsed_comp >= TGE_FIXED_ARRAY_SIZE(parsed_point.Components))
            {
                Tempest::Log(Tempest::LogLevel::Error, "too many components specified for point");
                return false;
            }
        }
        else if(c == ',')
        {
            result_vec.push_back(parsed_point);
            parsed_point = {};
            parsed_comp = 0;
            parsing = false;
        }
        else if(!isspace(c))
        {
            Tempest::Log(Tempest::LogLevel::Error, "invalid character in point declaration");
            return false;
        }
    }

    if(parsed_comp < TGE_FIXED_ARRAY_SIZE(parsed_point.Components) - 1 && parsing)
    {
        Tempest::Log(Tempest::LogLevel::Error, "incomplete number at point declaration");
        return false;
    }
    
    if(parsed_comp == TGE_FIXED_ARRAY_SIZE(parsed_point.Components) - 1)
    {
        result_vec.push_back(parsed_point);
    }

    *out_vec = std::move(result_vec);
    return true;
}

CommandLineOptsParser::CommandLineOptsParser(std::string name, bool unassociated_allowed)
    :   m_Name(name),
        m_UnassociatedAllowed(unassociated_allowed)
{
}

CommandLineOptsParser::~CommandLineOptsParser()
{
}

void CommandLineOptsParser::createOption(char short_name, std::string full_name, std::string description, bool has_arg, std::string default_value)
{
#ifndef NDEBUG
    for(auto& opt : m_Opts)
    {
        TGE_ASSERT(opt.second.getShortName() != short_name, "Overlapping options");
    }
#endif

    m_Opts[full_name] = OptionDescription(short_name, description, has_arg, default_value);
}

bool CommandLineOptsParser::parse(int argc, char* argv[])
{
    // TOODO: Better error reporting.
    for(int i = 1; i < argc; ++i)
    {
        TGE_ASSERT(argv[i], "Expected non-null command line option");
        char* opt = argv[i];
        if(opt[0] == '-' && !isdigit(opt[1]))
        {
            if(opt[1] == '-')
            {
                auto iter = m_Opts.find(opt + 2);
                if(iter == m_Opts.end())
                {
                    Log(LogLevel::Error, "Unknown command line option: " + std::string(opt));
                    return false;
                }
            }
            else if(isalpha(opt[1]))
            {
                char short_name = opt[1];
                if(opt[2] != '\0')
                {
                    Log(LogLevel::Error, "Short command line option must be made of a single character: " + std::string(opt));
                    return false;
                }
                auto iter = std::find_if(m_Opts.begin(), m_Opts.end(), [short_name](const OptionDescriptionMap::value_type& desc){ return short_name == desc.second.getShortName(); });
                if(iter == m_Opts.end())
                {
                    Log(LogLevel::Error, "Unknown command line option: " + std::string(opt));
                    return false;
                }
                if(!iter->second.getHasValue())
                {
                    iter->second.setValue("1");
                    continue;
                }
                if(++i == argc || (argv[i][0] == '-' && !isdigit(argv[i][1])))
                {
                    Log(LogLevel::Error, "Expected value after option: " + std::string(opt));
                    return false;
                }
                iter->second.setValue(argv[i]);
            }
            else
            {
                Log(LogLevel::Error, "Invalid command line option: " + std::string(opt));
                return false;
            }
        }
        else if(m_UnassociatedAllowed)
        {
            m_Unassociated.push_back(opt);
        }
        else
        {
            Log(LogLevel::Error, "Unexpected command line argument: " + std::string(opt));
            return false;
        }
    }
	return true;
}
    
void CommandLineOptsParser::printHelp(std::ostream& os)
{
    os << m_Name << " [OPTIONS] args\n"
       << "Usage:\n";

    auto size = m_Opts.size();

    std::vector<std::string> options_vec;
    size_t max_len = 0;

    for(auto i = m_Opts.begin(); i != m_Opts.end(); ++i)
    {
        std::stringstream ss;
        auto short_name = i->second.getShortName();
        ss << "\t";
        if(short_name != FullNameOnly)
            ss << "-" << short_name << ", ";
        ss << "--" << i->first;


        if(i->second.getHasValue())
        {
            auto value = i->second.getValue();
            if(!value.empty())
            {
                ss << "(=" << value << ")";
            }
        }

        ss << "    ";

        auto option_str = ss.str();
        max_len = std::max(option_str.length(), max_len);

        options_vec.push_back(option_str);
    }

    size_t idx = 0;
    for(auto i = m_Opts.begin(); i != m_Opts.end(); ++i)
    {
        os << std::setw(max_len) << std::left << options_vec[idx++] << i->second.getDescription() << "\n";
    }
    
}

template bool ParseCommaSeparatedVectors(const char* str, std::vector<Vector2>* out_vec);
template bool ParseCommaSeparatedVectors(const char* str, std::vector<Vector3>* out_vec);
}