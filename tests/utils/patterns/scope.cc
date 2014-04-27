#include "tempest/utils/testing.hh"
#include "tempest/utils/patterns.hh"

void TestAtScopeExit(bool& result)
{
    result = false;
    auto at_exit = Tempest::CreateAtScopeExit([&result]() { result = true; });
}

void TestRollback1(bool& result)
{
    result = true;
    auto transaction = Tempest::CreateTransaction([&result]() { result = true; });
    result = false;
}

void TestRollback2(bool& result)
{
    result = true;
    auto transaction = Tempest::CreateTransaction([&result]() { result = false; });
    transaction.commit();
}

void TestScoped(bool& result)
{
    result = false;
    auto scoped = Tempest::CreateScoped<bool*>([](bool* result) { *result = true; });
    scoped = &result;
}

TGE_TEST("Testing scoped templates and patterns.")
{
    bool result;
    TestAtScopeExit(result);
    TGE_ASSERT(result, "Scope exit does not actually trigger function when function gets terminated");
    
    TestRollback1(result);
    TGE_ASSERT(result, "Transaction does not perform rollback");
    TestRollback2(result);
    TGE_ASSERT(result, "Transaction does not commit changes");
    
    TestScoped(result);
    TGE_ASSERT(result, "Scoped does not clean up at the end of scope");
}