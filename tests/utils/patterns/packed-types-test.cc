#include "tempest/utils/testing.hh"
#include "tempest/utils/patterns.hh"

#include <memory>

struct TestData
{
    std::string       Subname;
    int                   Number;
};

class TestPacked
{
public:
    std::string       Name;
    PACKED_DATA(TestData) Pack;
    
    TestPacked(const TestPacked&)=delete;
    TestPacked& operator=(const TestPacked&)=delete;
    
private:
    TestPacked(size_t count, std::string name)
        :   Name(name),
            Pack(count) {}
     ~TestPacked()=default;
};

TGE_TEST("Testing for memory corruptions when using packed types")
{
    const char* names[] =
    {
        "First name",
        "Second name",
        "Third name"
    };
    
    const size_t name_count = TGE_FIXED_ARRAY_SIZE(names);
    
    std::string name("Some quite complicated pack");
    
    auto pack = Tempest::CreateScoped<TestPacked*>(Tempest::CreatePackedData<TestPacked>(name_count, name), [](TestPacked* pack) { Tempest::DestroyPackedData(pack); });
    
    for(size_t i = 0; i < name_count; ++i)
    {
        auto& pack_val = pack->Pack.Values[i];
        pack_val.Subname = names[i];
        pack_val.Number = i;
    }
    
    TGE_CHECK(pack->Name == name, "The pack name should be still the same; otherwise, data was overwritten when changing array values.");
    
    pack->Name = name;
    
    TGE_CHECK(pack->Pack.Count == name_count, "Pack size should not change ever.");
    
    for(size_t i = 0; i < name_count; ++i)
    {
        auto& pack_val = pack->Pack.Values[i];
        TGE_CHECK(pack_val.Subname == names[i], "Name should be the same. Adjacent cells should not affect each other.");
        TGE_CHECK(pack_val.Number == i, "Number should be also preserved. Adjacent cells should not affect each other.");
    }
    
    auto size1 = sizeof(pack->Name) + sizeof(pack->Pack); size1;
    auto size2 = sizeof(TestPacked); size2;
    
    TGE_CHECK(((char*)(pack->Pack.Values + pack->Pack.Count) - (char*)pack.get()) ==
                (sizeof(TestPacked) + name_count*sizeof(TestData)), "Invalid pack size");
}