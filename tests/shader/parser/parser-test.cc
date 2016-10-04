#include "tempest/utils/testing.hh"
#include "tempest/shader/shader-driver.hh"

TGE_TEST("Testing parser basic functionalities")
{
    std::string name = "main";
    auto _node = Tempest::AST::CreateNode<Tempest::Shader::FunctionSet>(TGE_DEFAULT_LOCATION, name);
    TGE_CHECK(_node.getNodeName() == "main", "Hm, our function for creating stuff is useless");
    
    Tempest::Shader::Driver effect_driver(nullptr);
    auto func_set = effect_driver.createStackNode<Tempest::Shader::FunctionSet>(TGE_DEFAULT_LOCATION, name);
    TGE_CHECK(func_set->getNodeName() == "main", "When we create a node we expect it to have what we have passed in");
}