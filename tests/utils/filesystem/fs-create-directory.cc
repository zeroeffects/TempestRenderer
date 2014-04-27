#include "tempest/utils/testing.hh"
#include "tempest/utils/file-system.hh"

#include <cstdio>

// What we test in this test case scenario is whether
// the polling service detects changes when new directories
// get created or files get modified within the monitored directory.

TGE_TEST("Testing file system library capabilities.")
{   
    // Lets create our own directory to prevent spurious triggering of some asserts.
#define TEST_DIR Tempest_TESTS_DIR "/test"
    auto ret = Tempest::Directory::mkdir(TEST_DIR);
    TGE_ASSERT(ret != Tempest::TGE_FS_ERROR, "Failed to create test directory.");

    Tempest::Directory::rmdir(TEST_DIR "/testdir");
    
    Tempest::FSPollingService poller;
    
    auto status = poller.initPollingService();
    TGE_ASSERT(status, "Polling service cannot be initialized.");
    
    status = poller.addWatch(Tempest::Path(TEST_DIR));
    TGE_ASSERT(status, "Cannot set watch for TEST_DIR");
    
    ret = Tempest::Directory::mkdir(TEST_DIR "/testdir");
    TGE_ASSERT(ret != Tempest::TGE_FS_EXISTS, "Directory was just deleted. It should not exist.");
    
    Tempest::FSEvents evts;
    
    status = poller.poll(evts);
    TGE_ASSERT(status, "Polling has failed");
    
    TGE_ASSERT(evts.size() == 1, "Single event expected for directory creation.");
    TGE_ASSERT(evts[0].type & Tempest::TGE_FS_EVENT_CREATE, "Unmatching events.");
    TGE_ASSERT(evts[0].name == "testdir", "Unmatching directory name."
              "Are you sure that you have not created another directory at the same time?");

    const char* filename = TEST_DIR "/test.txt";
    std::fstream fs(TEST_DIR "/test.txt", std::ios::out);
    TGE_ASSERT(fs.is_open(), "Could not open file writing");
    fs << "Hello world!" << std::endl;
    fs.close();
    
    auto err = remove(filename);
    TGE_ASSERT(!err, "Failed to remove file.");
    
    status = poller.poll(evts);
    TGE_ASSERT(status, "Polling has failed");
    
    TGE_ASSERT(evts.size() == 3, "Unexpected number of events.");
    
    status = poller.poll(evts);
    TGE_ASSERT(status, "Polling has failed");
    
    TGE_ASSERT(evts.empty(), "File system event encountered without anything being done by the application");

    status = poller.removeWatch(Tempest::Path(TEST_DIR));
    TGE_ASSERT(status, "Cannot set watch for TEST_DIR");
}