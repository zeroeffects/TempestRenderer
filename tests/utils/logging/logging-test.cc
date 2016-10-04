#include "tempest/utils/testing.hh"
#include "tempest/utils/assert.hh"
#include "tempest/utils/logging.hh"

#include <algorithm>
#include <fstream>
#include <iterator>

using namespace Tempest;

TGE_TEST("Tests for proper execution of the logging functionality")
{
    Log(LogLevel::Info, "That's testing whether the standard output works without crashing");

    std::string app_str("Hello in synchronous mode");    
    
    {
        LogFile log("test-log.txt");
        
        // Test whether it always output in synchronous mode.
        Log(LogLevel::Error, app_str);
        
        auto str = LogFile::read();
        
        auto idx = str.find("standard output");
        TGE_CHECK(idx == std::string::npos, "The previous message should not appear. It was printed to the standard output only.");
        
        idx = str.find(app_str);
        TGE_CHECK(idx != std::string::npos, "This message should appear because it was also appended to the log file.");
    }
    
    std::fstream fs("test-log.txt", std::ios::in);
        std::istreambuf_iterator<char> start_iter(fs.rdbuf());
        std::istreambuf_iterator<char> end_iter;
        auto iter = std::search(start_iter, end_iter, app_str.begin(), app_str.end());
        TGE_CHECK(iter != end_iter, "Logged string not found");
    fs.close();
}