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

%skeleton "../parser/lalr1-tge-ext.cc"
%require "2.4"
%define parser_class_name "Parser"
%define namespace "Tempest::Shader"

%code requires
{
#include "tempest/utils/logging.hh"
#include "tempest/shader/shader-driver.hh"

#define YYSTYPE Tempest::AST::UnmanagedNode
}

%name-prefix="shader_"
%parse-param { Driver& driver }
%lex-param { Driver& driver }

%locations
%initial-action
{
    @$.begin.filename = @$.end.filename = &driver.__FileName;
};

//%debug
%error-verbose

%token                  T_END                   0   "end of file"
%token                  T_TYPE_VOID                 "void"
%token                  T_ADD_ASSIGN                "+="
%token                  T_SUB_ASSIGN                "-="
%token                  T_MUL_ASSIGN                "*="
%token                  T_DIV_ASSIGN                "/="
%token                  T_BITWISE_MOD_ASSIGN        "%="
%token                  T_BITWISE_AND_ASSIGN        "&="
%token                  T_BITWISE_XOR_ASSIGN        "^="
%token                  T_BITWISE_OR_ASSIGN         "|="
%token                  T_OR                        "||"
%token                  T_AND                       "&&"
%token                  T_XOR                       "^^"
%token                  T_SHIFT_RIGHT               ">>"
%token                  T_SHIFT_LEFT                "<<"
%token                  T_EQUAL                     "=="
%token                  T_NEQUAL                    "!="
%token                  T_LEQUAL                    "<="
%token                  T_GEQUAL                    ">="
%token                  T_INCR                      "++"
%token                  T_DECR                      "--"

%token                  T_STRUCT_QUALIFIER          "struct qualifier"
%token                  T_IMPORT                    "import"
%token                  T_IF                        "if"
%token                  T_ELSE                      "else"
%token                  T_BREAK                     "break"
%token                  T_CONTINUE                  "continue"
%token                  T_RETURN                    "return"
%token                  T_DO                        "do"
%token                  T_FOR                       "for"
%token                  T_WHILE                     "while"
%token                  T_SWITCH                    "switch"
%token                  T_CASE                      "case"
%token                  T_DEFAULT                   "default"
%token <TypeRef>        T_INTEGER                   "integer"
%token <TypeRef>        T_UNSIGNED                  "unsigned integer"
%token <TypeRef>        T_FLOAT                     "float"
%token <TypeRef>        T_BOOLEAN                   "boolean"
%token <StringLiteral>  T_STRING_LITERAL            "string literal"
%token                  T_PASS                      "pass"
%token                  T_TECHNIQUE                 "technique"
%token                  T_VERTEX_QUALIFIER          "vertex qualifier"
%token                  T_GEOMETRY_QUALIFIER        "geometry qualifier"
%token                  T_FRAGMENT_QUALIFIER        "fragment qualifier"
%token                  T_SHADER                    "shader"
%token <TypeRef>        T_TYPE                      "type"
%token <FunctionSetRef> T_FUNCTION                  "function"
%token <VariableRef>    T_VARIABLE                  "variable"
%token                  T_LAYOUT_QUALIFIER          "layout qualifier"
%token                  T_INVARIANT_QUALIFIER       "invariant qualifier"
%token                  T_CONST_QUALIFIER           "const qualifier"
%token                  T_CENTROID_QUALIFIER        "centroid qualifier"
%token                  T_SAMPLE_QUALIFIER          "sample qualifier"
%token                  T_IN_QUALIFIER              "in qualifier"
%token                  T_OUT_QUALIFIER             "out qualifier"
%token                  T_INOUT_QUALIFIER           "inout qualifier"
%token                  T_HIGHP_QUALIFIER           "highp qualifier"
%token                  T_MEDIUMP_QUALIFIER         "mediump qualifier"
%token                  T_LOWP_QUALIFIER            "lowp qualifier"
%token                  T_FLAT_QUALIFIER            "flat qualifier"
%token                  T_SMOOTH_QUALIFIER          "smooth qualifier"
%token                  T_NOPERSPECTIVE_QUALIFIER   "noperspective qualifier"
%token <Identifier>     T_IDENTIFIER                "identifier"
%token                  T_BUFFER_QUALIFIER          "buffer qualifier"
%token                  T_CONSTANT_QUALIFIER        "constant qualifier"
%token                  T_UNIFORM_QUALIFIER         "uniform qualifier"
%token                  T_RESOURCE_QUALIFIER        "resource qualifier"
%token                  T_STRUCTBUFFER_QUALIFIER    "structbuffer qualifier"

%type <List>                        translation_unit shader_body technique_body pass_body function_variables_list statement_list switch_statement_list
%type <List>                        layout_id_list layout_header definitions_block definitions_list function_variables_non_empty_list
%type <List>                        function_arg_list buffer_list struct_body
%type <void>                        external_declaration statement iteration_statement for_init_statement block_statement layout_id else_statement
%type <void>                        selection_statement switch_statement case_statement default_statement expression_statement jump_statement
%type <void>                        definition_pair definition_value effect_file
%type <Buffer>                      buffer buffer_declaration
%type <Import>                      import
%type <TypeRef>                     shader struct_declaration
%type <Technique>                   technique
%type <Pass>                        pass
%type <VariableRef>                 variable output_variable interpolation_variable invariant_variable const_variable variable_with_layout
%type <VariableRef>                 function_variable gvariable buffer_variable
%type <void>                        function
%type <IntermFuncNode>              function_declaration function_definition function_statement
%type <DeclarationInfo>             function_header
%type <FuncDeclarationInfo>         declared_function_header
%type <TypeRef>                     return_type
%type <Expression>                  expression assignment_expression variable_expression conditional_expression logical_or_expression logical_xor_expression
%type <Expression>                  bitwise_and_expression bitwise_or_expression bitwise_xor_expression bitwise_shift_expression equality_expression
%type <Expression>                  prefix_expression suffix_expression multiplicative_expression additive_expression parentheses_expression
%type <Expression>                  condition conditionopt  logical_and_expression scalar_expression function_call relational_expression
%type <Expression>                  optional_expression
%type <VarDeclList>                 variable_declaration basic_variable_declaration
%type <Declaration>                 declaration_statement
%type <InvariantDeclaration>        invariant_declaration
%type <ValueShaderType>             shader_type

//%printer { if($$) debug_stream() << $$->getValue(); } "identifier"
//%printer { if($$) debug_stream() << $$->getNodeName(); } "variable" "function" "type"

%destructor { $$.destroy(); } <*>

%start effect_file

%code provides
{
#define YY_DECL Tempest::Shader::Parser::token_type shader_lex(YYSTYPE* yylval, \
                                                               Tempest::Shader::Parser::location_type* yylloc, \
                                                               Tempest::Shader::Driver& driver)
YY_DECL;

namespace Tempest
{
namespace Shader
{
NodeT<Expression> CreateBinaryOperator(Driver& driver, Location loc, BinaryOperatorType _type,
                                          NodeT<Expression> left_expr, NodeT<Expression> right_expr);
NodeT<Expression> CreateUnaryOperator(Driver& driver, Location loc, UnaryOperatorType _type,
                                         NodeT<Expression> expr);
void ErrorRedefinition(Driver& driver, Location loc, NodeT<VariableRef> var);

inline Location ToLocation(const location& loc) { return Location{ loc.begin.filename, loc.begin.line, loc.begin.column }; }
}
}
}

%%

effect_file
    : translation_unit                                      { driver.setASTRoot($1); $$ = AST::Node(); }
    ;

translation_unit
    : /* empty */                                           { $$ = NodeT<List>(); }
    | external_declaration translation_unit                 { $$ = CreateNodeTyped<List>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

external_declaration
    : import                                                { $$ = $1; }
    | shader                                                { $$ = $1; }
    | technique                                             { $$ = $1; }
    | function                                              { $$ = $1; } // Some shared stuff between the shaders
    | buffer                                                { $$ = $1; }
    | struct_declaration                                    { $$ = $1; }
    ;


buffer
    : "constant qualifier" buffer_declaration               { auto buf = $2; buf->setBufferType(BufferType::Constant); $$ = std::move(buf); }
    | buffer_declaration                                    { $$ = $1; }
    | "resource qualifier" buffer_declaration               { auto buf = $2; buf->setBufferType(BufferType::Resource); $$ = std::move(buf); }
    // It is far from being an actual buffer. More of linear memory with externally defined bounds.
    | "structbuffer qualifier" "type" "identifier" ';'      {
                                                                auto type = $2;
                                                                auto identifier = $3;
                                                                TGE_ASSERT(type && identifier, "Valid type and identifier expected. Potential lexer bug");
                                                                AST::NodeT<VariableRef> var;
                                                                if(type->getTypeEnum() == Shader::ElementType::Struct)
                                                                {
                                                                    auto arr_type = driver.createInternalType<ArrayType>(ToLocation(@$), type.get(), AST::Node());
                                                                    var = driver.createStackNode<Variable>(ToLocation(@$), arr_type.get(), identifier->getValue());
                                                                    var->setStorage(StorageQualifier::StructBuffer);
                                                                }
                                                                else
                                                                {
                                                                    driver.error(ToLocation(@$), type->getNodeName() + " is not struct type. structbuffer is limited to struct types only.");
                                                                }
                                                                $$ = CreateNode<Declaration>(ToLocation(@$), std::move(var));
                                                            }
    | "structbuffer qualifier" "type" "variable" ';'        { ErrorRedefinition(driver, ToLocation(@$), $3); $2; $$ = NodeT<Declaration>(); }
    ;

buffer_declaration
    : "buffer qualifier" "identifier" '{'
          buffer_list
       '}'                                                  {
                                                                auto identifier = $2;
                                                                $$ = CreateNode<Buffer>(ToLocation(@$), identifier->getValue(), $4);
                                                            }
    ;
    
buffer_list
    : /* empty */                                           { $$ = NodeT<List>(); }
    | buffer_variable buffer_list                           { $$ = CreateNode<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

buffer_variable
    : layout_header variable ';'                            { auto var = $2; if(var) var->setLayout($1); $$ = std::move(var); }
    | variable ';'                                          { $$ = CreateNode<Declaration>(ToLocation(@$), $1); }
    ;

import
    : "import" "string literal" definitions_block ';'       { $$ = CreateNode<Import>(ToLocation(@$), "\"" + $2->getValue() + "\"", $3); }
    ;

definitions_block
    : /* empty */                                           { $$ = NodeT<List>(); }
    | '{' definitions_list '}'                              { $$ = $2; }
    ;

definitions_list
    : /* empty */                                           { $$ = NodeT<List>(); }
    | definition_pair definitions_list                      { $$ = CreateNode<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

definition_pair
    : "identifier" '=' definition_value ';'                 { $$ = CreateNode<BinaryOperator>(ToLocation(@$), TGE_EFFECT_ASSIGN, $1, $3); }
    ;

definition_value
    : "integer"                                             { $$ = $1; }
    | "unsigned integer"                                    { $$ = $1; }
    | "float"                                               { $$ = $1; }
    | "boolean"                                             { $$ = $1; }
    ;

shader
    : shader_type "identifier" '{'
            shader_body
    '}'                                                     {
                                                                driver.endShader();
                                                                auto shader_type = $1;
                                                                auto identifier = $2;
                                                                TGE_ASSERT(shader_type, "Valid shader qualifier expected");
                                                                TGE_ASSERT(identifier, "Valid identifier expected. Potential lexer bug.");
                                                                $$ = driver.createStackType<ShaderDeclaration>(ToLocation(@$), shader_type->getValue(), identifier ? identifier->getValue() : string(), $4);
                                                            }
    ;

shader_type
    : "vertex qualifier" "shader"                           { driver.beginShader(ShaderType::VertexShader); $$ = CreateNode<Value<ShaderType>>(ToLocation(@$), ShaderType::VertexShader); }
    | "fragment qualifier" "shader"                         { driver.beginShader(ShaderType::FragmentShader); $$ = CreateNode<Value<ShaderType>>(ToLocation(@$), ShaderType::FragmentShader); }
    ;

shader_body
    :   /* empty */                                         { $$ = NodeT<List>(); }
    | gvariable shader_body                                 { $$ = CreateNode<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, CreateNode<Declaration>(ToLocation(@$), $1), $2); }
    | function shader_body                                  { $$ = CreateNode<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    | invariant_declaration ';'  shader_body                { $$ = CreateNode<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $3); }
    | struct_declaration shader_body                        { $$ = CreateNode<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

technique
    : "technique" "identifier" '{'                          { driver.beginTechnique(); }
            technique_body
    '}'                                                     {
                                                                driver.endTechnique();
                                                                auto identifier = $2;
                                                                TGE_ASSERT(identifier, "Valid identifier expected. Potential lexer bug.");
                                                                $$ = driver.createStackNode<Technique>(ToLocation(@$), identifier->getValue(), $5);
                                                            }
    ;

technique_body
    : /* empty */                                           { $$ = NodeT<List>(); }
    | pass technique_body                                   { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

pass
    : "pass" "identifier" '{'                               { driver.beginBlock(); }
            pass_body
    '}'                                                     {
                                                                driver.endBlock();
                                                                auto identifier = $2;
                                                                TGE_ASSERT(identifier, "Valid identifier expected. Potential lexer bug.");
                                                                $$ = driver.createStackNode<Pass>(ToLocation(@$), identifier->getValue(), $5);
                                                            }
    ;

pass_body
    : /* empty */                                           { $$ = NodeT<List>(); }
    | function_call ';' pass_body                           { auto expr = $1; $$ = CreateNode<List>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, expr ? std::move(expr->getSecond()) : AST::Node(), $3); }
    ;

function
    : function_statement                                    { $$ = std::move($1->getFirst()); }
    ;
    
function_statement
    : function_declaration                                  { $$ = $1; }
    | function_definition                                   { $$ = $1; }
    ;
    
struct_declaration
    : "struct qualifier" "identifier" '{'                   { driver.beginBlock(); }
        struct_body
      '}'';'                                                {
                                                                driver.endBlock();
                                                                auto identifier = $2;
                                                                $$ = CreateNode<Declaration>(ToLocation(@$), driver.createStackType<StructType>(ToLocation(@$), identifier ? identifier->getValue() : string(), $5));
                                                            }
    ;

struct_body
    : /* empty */                                           { $$ = AST::NodeT<List>(); }
    | basic_variable_declaration ';' struct_body            {
                                                                auto decl = $1;
                                                                AST::Node result;
                                                                if(decl && decl->getSecond())
                                                                {
                                                                    // Reduce nodes by removing lists of a single node and placing the node directly in the AST
                                                                    AST::Node moved_decl;
                                                                    if(*decl->getSecond()->next())
                                                                        moved_decl = std::move(decl->getSecond());
                                                                    else
                                                                        moved_decl = std::move(*decl->getSecond()->current_front());
                                                                    result = CreateNode<Declaration>(ToLocation(@$), std::move(moved_decl));
                                                                }
                                                                $$ = CreateNodeTyped<ListElement>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, std::move(result), $3);
                                                            }
    ;

function_declaration
    : function_header function_variables_list ')' ';'       {
                                                                driver.endBlock();
                                                                auto func_decl_info = $1;
                                                                auto func_var_list = $2;
                                                                NodeT<Reference<FunctionDeclaration>> result;
                                                                FunctionDeclaration* func_decl_ptr = nullptr;
                                                                if(func_decl_info && func_decl_info->getSecond())
                                                                {
                                                                    string func_name = func_decl_info->getSecond()->getValue();
                                                                    auto func_decl = CreateNodeTyped<FunctionDeclaration>(ToLocation(@$), func_decl_info->getFirst(),
                                                                                                                          func_name,
                                                                                                                          std::move(func_var_list));
                                                                    func_decl_ptr = func_decl.get();
                                                                    result = CreateNode<Reference<FunctionDeclaration>>(ToLocation(@$), func_decl.get());
                                                                    auto func_set = driver.createStackNode<FunctionSet>(ToLocation(@$), func_name);
                                                                    func_set->pushFunction(std::move(func_decl));
                                                                }
                                                                $$ = CreateNode<IntermFuncNode>(ToLocation(@$), std::move(result), func_decl_ptr);
                                                            }
    | declared_function_header
    	  function_variables_list ')' ';'                   {
                                                                driver.endBlock();
                                                                NodeT<Reference<FunctionDeclaration>> func;
                                                                FunctionDeclaration* func_decl_ptr = nullptr;
                                                                auto func_decl_info = $1;
                                                                auto arg_list = $2;
                                                                if(func_decl_info)
                                                                {
                                                                    func_decl_ptr = func_decl_info->getSecond()->getFunction(arg_list.get());
                                                                    if(!func_decl_ptr)
                                                                    {
                                                                        auto func_decl_node = CreateNodeTyped<FunctionDeclaration>(ToLocation(@$), func_decl_info->getFirst(), func_decl_info->getSecond()->getNodeName(), std::move(arg_list));
                                                                        func_decl_ptr = func_decl_node.get();
                                                                        func_decl_info->getSecond()->pushFunction(std::move(func_decl_node));
                                                                    }
                                                                    func = CreateNode<Reference<FunctionDeclaration>>(ToLocation(@$), func_decl_ptr);
                                                                }
                                                                $$ = CreateNode<IntermFuncNode>(ToLocation(@$), std::move(func), func_decl_ptr);
                                                            }
    ;
    
function_definition
    : function_header function_variables_list ')' '{'   
             statement_list
      '}'                                                   {
                                                                driver.endBlock();
                                                                auto func_decl_info = $1;
                                                                auto arg_list = $2;
                                                                NodeT<FunctionDefinition> func;
                                                                FunctionDeclaration* func_decl_ptr = nullptr;
                                                                if(func_decl_info && func_decl_info->getSecond())
                                                                {
                                                                    string func_name = func_decl_info->getSecond()->getValue();
                                                                    auto func_decl = CreateNodeTyped<FunctionDeclaration>(ToLocation(@$), func_decl_info->getFirst(),
                                                                                                                          func_name,
                                                                                                                          std::move(arg_list));
                                                                    func = CreateNodeTyped<FunctionDefinition>(ToLocation(@$), func_decl.get(), $5);
                                                                    func_decl_ptr = func_decl.get();
                                                                    auto func_set = driver.createStackNode<FunctionSet>(ToLocation(@$), func_name);
                                                                    func_set->pushFunction(std::move(func_decl));
                                                                }
                                                                $$ = CreateNode<IntermFuncNode>(ToLocation(@$), std::move(func), func_decl_ptr);
                                                            }
    | declared_function_header
    	  function_variables_list ')' '{'
             statement_list
      '}'                                                   {
                                                                driver.endBlock();
                                                                auto func_decl_info = $1;
                                                                auto arg_list = $2;
                                                                NodeT<FunctionDefinition> func;
                                                                FunctionDeclaration* func_decl_ptr = nullptr;
                                                                if(func_decl_info)
                                                                {
                                                                    func_decl_ptr = func_decl_info->getSecond()->getFunction(arg_list.get());
                                                                    if(!func_decl_ptr)
                                                                    {
                                                                        auto func_decl_node = CreateNodeTyped<FunctionDeclaration>(ToLocation(@$), func_decl_info->getFirst(), func_decl_info->getSecond()->getNodeName(), std::move(arg_list));
                                                                        func_decl_ptr = func_decl_node.get();
                                                                        func_decl_info->getSecond()->pushFunction(std::move(func_decl_node));
                                                                    }
                                                                    
                                                                    func = CreateNode<FunctionDefinition>(ToLocation(@$), func_decl_ptr, $5);
                                                                }

                                                                $$ = CreateNode<IntermFuncNode>(ToLocation(@$), std::move(func), func_decl_ptr);
                                                            }
    ;

function_header
    : "type" "identifier" '('                               { auto _type = $1; driver.beginBlock(); $$ = CreateNode<DeclarationInfo>(ToLocation(@$), _type.get(), $2); }
    | "void" "identifier" '('                               { driver.beginBlock(); $$ = CreateNode<DeclarationInfo>(ToLocation(@$), nullptr, $2); }
    ;

declared_function_header
	: return_type "function" '('                            { 
                                                                driver.beginBlock();
                                                                auto _type = $1;
                                                                auto function = $2;
                                                                TGE_ASSERT(function, "Valid function name expected");
                                                                $$ = CreateNode<FuncDeclarationInfo>(ToLocation(@$), _type.get(), function.get());
                                                            }
	;

return_type
    : "void"                                                { $$ = NodeT<Type>(); }
    | "type"                                                { $$ = $1; }
    ;

statement_list
    : /* empty */                                           { $$ = NodeT<List>(); }
    | statement statement_list                              { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

statement
    : declaration_statement                                 { $$ = $1; }
    | selection_statement                                   { $$ = $1; }
    | switch_statement                                      { $$ = $1; }
    | iteration_statement                                   { $$ = $1; }
    | expression_statement                                  { $$ = $1; }
    | block_statement                                       { $$ = $1; }
    | jump_statement                                        { $$ = $1; }
    ;

iteration_statement
    : "do"                                                  { driver.beginBlock(); }
    
            statement "while" '(' expression ')' ';'        {
                                                                driver.endBlock();
                                                                auto loop_cond = $6;
                                                                $$ = CreateNode<WhileStatement>(ToLocation(@$), loop_cond ? std::move(loop_cond->getSecond()) : AST::Node(), $3, true);
                                                            }
    | "for" '('                                             { driver.beginBlock(); }
                for_init_statement
                conditionopt ';' expression ')' statement   {
                                                                driver.endBlock();
                                                                auto loop_cond = $5;
                                                                auto loop_iter = $7;
                                                                $$ = CreateNode<ForStatement>(ToLocation(@$), $4, loop_cond ? std::move(loop_cond->getSecond()) : AST::Node(),
                                                                                              loop_iter ? std::move(loop_iter->getSecond()) : AST::Node(), $9);
                                                            }
    | "while" '('                                           { driver.beginBlock(); }
                  condition ')' statement                   {
                                                                driver.endBlock();
                                                                auto loop_cond = $4;
                                                                $$ = CreateNode<WhileStatement>(ToLocation(@$), loop_cond ? std::move(loop_cond->getSecond()) : AST::Node(), $6);
                                                            }
    ;

for_init_statement
    : declaration_statement                                 { $$ = $1; }
    | expression_statement                                  { $$ = $1; }
    ;

conditionopt
    : /* empty */                                           { $$ = CreateNode<Expression>(ToLocation(@$), nullptr, AST::Node()); }
    | condition                                             { $$ = $1; }
    ;

condition
    : expression                                            { $$ = $1; }
    | variable '=' conditional_expression                   {
                                                                auto rvalue = $3;
                                                                auto lvalue = $1;
                                                                
                                                                auto* lvalue_type = lvalue ? lvalue->getType() : nullptr;
                                                                
                                                                auto decl = CreateNode<Declaration>(ToLocation(@$), CreateNode<BinaryOperator>(ToLocation(@$), TGE_EFFECT_ASSIGN, std::move(lvalue),
                                                                                                                                   rvalue ? std::move(rvalue->getSecond()) : AST::Node()));
                                                                $$ = CreateNode<Expression>(ToLocation(@$), lvalue_type, std::move(decl));
                                                            }
    ;

jump_statement
    : "continue" ';'                                        { $$ = CreateNode<JumpStatement>(ToLocation(@$), JumpStatementType::Continue); }
    | "break" ';'                                           { $$ = CreateNode<JumpStatement>(ToLocation(@$), JumpStatementType::Break); }
    | "return" ';'                                          { $$ = CreateNode<ReturnStatement>(ToLocation(@$)); }
    | "return" expression ';'                               { 
                                                                auto expr = $2;
                                                                $$ = CreateNode<ReturnStatement>(ToLocation(@$), expr ? std::move(expr->getSecond()) : AST::Node());
                                                            }
    ;

selection_statement
    : "if" '(' expression ')'                               { driver.beginBlock(); }
            statement                                       { driver.endBlock();  }
      else_statement                                        {
                                                                auto cond_expr = $3;
                                                                $$ = CreateNode<IfStatement>(ToLocation(@$), cond_expr ? std::move(cond_expr->getSecond()) : AST::Node(), $6, $8);
                                                            }
    ;

else_statement
    : /* empty */                                           { $$ = AST::Node(); }
    | "else"                                                { driver.beginBlock(); }
            statement                                       { driver.endBlock(); $$ = $3; }
    ;


switch_statement
    : "switch" '(' expression ')' '{'                       { driver.beginBlock(); }
        switch_statement_list
    '}'                                                     {
                                                                driver.endBlock();
                                                                auto cond_expr = $3;
                                                                $$ = CreateNode<IfStatement>(ToLocation(@$), cond_expr ? std::move(cond_expr->getSecond()) : AST::Node(), $7);
                                                            }
    ;

switch_statement_list
    : /* empty */                                           { $$ = NodeT<List>(); }
    | default_statement                                     { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, NodeT<List>()); }
    | case_statement switch_statement_list                  { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_SEMICOLON_SEPARATED_LIST, $1, $2); }
    ;

case_statement
    : "case" expression ':'                                 { driver.beginBlock(); }
            statement_list                                  {
                                                                driver.endBlock();
                                                                auto case_expr = $2;
                                                                $$ = CreateNode<CaseStatement>(ToLocation(@$), case_expr ? std::move(case_expr->getSecond()) : AST::Node(), $5);
                                                            }
    ;

default_statement
    : "default" ':'                                         { driver.beginBlock(); }
            statement_list                                  {
                                                                driver.endBlock();
                                                                $$ = CreateNode<CaseStatement>(ToLocation(@$), AST::Node(), $4);
                                                            }
    ;

expression_statement
    : expression ';'                                        {
                                                                auto expr = $1;
                                                                $$ = expr ? std::move(expr->getSecond()) : AST::Node();
                                                            }
    ;

declaration_statement
    : variable_declaration ';'                              {
                                                                auto decl = $1;
                                                                AST::Node result;
                                                                if(decl && decl->getSecond())
                                                                {
                                                                    // Reduce nodes by removing lists of a single node and placing the node directly in the AST
                                                                    AST::Node moved_decl;
                                                                    if(*decl->getSecond()->next())
                                                                        moved_decl = std::move(decl->getSecond());
                                                                    else
                                                                        moved_decl = std::move(*decl->getSecond()->current_front());
                                                                    result = CreateNode<Declaration>(ToLocation(@$), std::move(moved_decl));
                                                                }
                                                                $$ = std::move(result);
                                                            }
    ;

variable_declaration
    : variable_declaration '=' conditional_expression       {
                                                                auto var_decl = $1;
                                                                auto rvalue = $3;
                                                                if(var_decl && var_decl->getSecond())
                                                                {
                                                                    auto* ptr = var_decl->getSecond()->back();
                                                                    // Here is the deal. We have acquired the actual variable in some of the previous steps.
                                                                    // However, suddenly we have an assignment. We want to move it to the new assignment initialization
                                                                    // and move the whole operation in its place.
                                                                    *ptr = CreateNode<BinaryOperator>(ToLocation(@$), TGE_EFFECT_ASSIGN, std::move(*ptr), rvalue ? std::move(rvalue->getSecond()) : AST::Node());
                                                                }
                                                                $$ = std::move(var_decl);
                                                            }
    | basic_variable_declaration                            { $$ = $1; }
    ;

basic_variable_declaration
    : variable_declaration ',' "identifier"                 {
                                                                auto decl = $1;
                                                                auto identifier = $3;
                                                                TGE_ASSERT(identifier, "Expecting valid identifier.");
                                                                if(decl && decl->getFirst() && decl->getSecond())
                                                                {
                                                                    auto var = driver.createStackNode<Variable>(ToLocation(@$), decl->getFirst(), identifier->getValue());
                                                                    decl->getSecond()->push_back(std::move(var));
                                                                }
                                                                $$ = std::move(decl);
                                                            }
    | variable_declaration ',' "variable"                   { ErrorRedefinition(driver, ToLocation(@$), $3); $1; $$ = NodeT<Variable>(); }
    | variable_declaration ',' "identifier" '[' optional_expression ']'
                                                            {
                                                                auto var_decl = $1;
                                                                auto identifier = $3;
                                                                auto arr_size = $5;
                                                                TGE_ASSERT(identifier, "Valid identifier expected. Potential lexer bug.");
                                                                if(identifier && var_decl->getFirst() && var_decl->getSecond())
                                                                {
                                                                    auto arr_type = driver.createInternalType<ArrayType>(ToLocation(@$), var_decl ? var_decl->getFirst() : nullptr,
                                                                                                                         arr_size ? std::move(arr_size->getSecond()) : AST::Node());
                                                                    auto var = driver.createStackNode<Variable>(ToLocation(@$), arr_type.get(), identifier->getValue());
                                                                    var_decl->getSecond()->push_back(std::move(var));
                                                                }
                                                                $$ = std::move(var_decl);
                                                            }
    | variable_declaration ',' "variable" '[' optional_expression ']' { ErrorRedefinition(driver, ToLocation(@$), $3); $1; $5; $$ = NodeT<Variable>(); }
    | variable                                              {
                                                                auto var = $1;
                                                                const Type* _type = var->getType();
                                                                $$ = var ? CreateNodeTyped<Expression>(ToLocation(@$), _type, CreateNode<ListElement>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, std::move(var))) : NodeT<Expression>();
                                                            }
    ;

optional_expression
    : /* empty */                                           { $$ = CreateNodeTyped<Expression>(ToLocation(@$), nullptr, AST::Node()); }
    | expression                                            { $$ = $1; }
    ;
    
block_statement
    : '{'                                                   { driver.beginBlock(); }
            statement_list
    '}'                                                     {
                                                                driver.endBlock();
                                                                $$ = CreateNode<Block>(ToLocation(@$), $3);
                                                            }
    ;

gvariable
    : variable ';'                                          { $$ = $1; }
    | output_variable ';'                                   { $$ = $1; }
    | interpolation_variable ';'                            { $$ = $1; }
    | invariant_variable ';'                                { $$ = $1; }
    | const_variable ';'                                    { $$ = $1; }
    | variable_with_layout ';'                              { $$ = $1; }
    ;

invariant_declaration
    : "invariant qualifier" "variable"                      { $$ = CreateNode<InvariantDeclaration>(ToLocation(@$), $2.get()); }
    ;

variable_with_layout
    : layout_header output_variable                         { auto var = $2; if(var) var->setLayout($1); $$ = std::move(var); }
    | layout_header interpolation_variable                  { auto var = $2; if(var) var->setLayout($1); $$ = std::move(var); }
    ;

layout_header
    : "layout qualifier" '(' layout_id_list ')'             { $$ = $3; }
    ;

layout_id_list
    : layout_id ',' layout_id_list                          { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, $1, $3);}
    | layout_id                                             { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, $1, NodeT<List>()); }
    ;

layout_id
    : "identifier"                                          { $$ = $1; }
    | "identifier" '=' "identifier"                         { $$ = CreateNode<BinaryOperator>(ToLocation(@$), TGE_EFFECT_ASSIGN, $1, $3); }
    | "identifier" '=' "integer"                            { $$ = CreateNode<BinaryOperator>(ToLocation(@$), TGE_EFFECT_ASSIGN, $1, $3); }
    ;

invariant_variable
    : "invariant qualifier" variable                        { auto var = $2; if(var) var->setInvariant(true); $$ = std::move(var); }
    | "invariant qualifier" output_variable                 { auto var = $2; if(var) var->setInvariant(true); $$ = std::move(var); }
    | "invariant qualifier" interpolation_variable          { auto var = $2; if(var) var->setInvariant(true); $$ = std::move(var); }
    ;

interpolation_variable
    : "flat qualifier" output_variable                      { auto var = $2; if(var) var->setInterpolation(InterpolationQualifier::Flat); $$ = std::move(var); }
    | "noperspective qualifier" output_variable             { auto var = $2; if(var) var->setInterpolation(InterpolationQualifier::Noperspective); $$ = std::move(var); }
    | "smooth qualifier" output_variable                    { auto var = $2; if(var) var->setInterpolation(InterpolationQualifier::Smooth); $$ = std::move(var); }
    ;

const_variable
    : "const qualifier" variable                            { auto var = $2; if(var) var->setStorage(StorageQualifier::Const); $$ = std::move(var); }
    ;

output_variable
    : "in qualifier" variable                               { auto var = $2; if(var) var->setStorage(StorageQualifier::In); $$ = std::move(var); }
    | "out qualifier" variable                              { auto var = $2; if(var) var->setStorage(StorageQualifier::Out); $$ = std::move(var); }
    | "centroid qualifier" "in qualifier" variable          { auto var = $3; if(var) var->setStorage(StorageQualifier::CentroidIn); $$ = std::move(var); }
    | "centroid qualifier" "out qualifier" variable         { auto var = $3; if(var) var->setStorage(StorageQualifier::CentroidOut); $$ = std::move(var); }
    | "sample qualifier" "in qualifier" variable            { auto var = $3; if(var) var->setStorage(StorageQualifier::SampleIn); $$ = std::move(var); }
    | "sample qualifier" "out qualifier" variable           { auto var = $3; if(var) var->setStorage(StorageQualifier::SampleOut); $$ = std::move(var); }
    ;

function_variables_list
    : /* empty */                                           { $$ = NodeT<List>(); }
    | function_variables_non_empty_list                     { $$ = $1; }
    ;

function_variables_non_empty_list
    : function_variable                                     { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, CreateNode<Declaration>(ToLocation(@$), $1), NodeT<List>()); }
    | function_variable ',' function_variables_list         { $$ = CreateNode<List>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, CreateNode<Declaration>(ToLocation(@$), $1), $3); }
    ;

function_variable
    : variable                                              { $$ = $1; }
    | "in qualifier" function_variable                      { auto var = $2; if(var) var->setStorage(StorageQualifier::In); $$ = std::move(var); }
    | "out qualifier" function_variable                     { auto var = $2; if(var) var->setStorage(StorageQualifier::Out); $$ = std::move(var); }
    | "inout qualifier" function_variable                   { auto var = $2; if(var) var->setStorage(StorageQualifier::InOut); $$ = std::move(var); }
    | "const qualifier" function_variable                   { auto var = $2; if(var) var->setStorage(StorageQualifier::Const); $$ = std::move(var); }
    ;

variable
    : "type" "identifier"                                   {
                                                                auto type = $1;
                                                                auto identifier = $2;
                                                                TGE_ASSERT(type && identifier, "Valid type and identifier expected. Potential lexer bug");
                                                                $$ = driver.createStackNode<Variable>(ToLocation(@$), type.get(), identifier->getValue());
                                                            }
    | "type" "variable"                                     { ErrorRedefinition(driver, ToLocation(@$), $2); $1; $$ = NodeT<Variable>(); }
    | "type" "variable" '[' optional_expression ']'         { ErrorRedefinition(driver, ToLocation(@$), $2); $1; $4; $$ = NodeT<Variable>(); }
    | "type" "identifier" '[' optional_expression ']'       {
                                                                auto type = $1;
                                                                auto identifier = $2;
                                                                auto expr = $4;
                                                                TGE_ASSERT(type && identifier, "Valid type and identifier expected. Potential lexer bug.");
                                                                auto arr_type = driver.createInternalType<ArrayType>(ToLocation(@$), type.get(), expr ? std::move(expr->getSecond()) : AST::Node());
                                                                $$ = driver.createStackNode<Variable>(ToLocation(@$), arr_type.get(), identifier->getValue());
                                                            }
    ;

expression
    : assignment_expression                                 { $$ = $1; }
    | expression ',' assignment_expression                  { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_COMMA, $1, $3); }
    ;

assignment_expression
    : conditional_expression                                { $$ = $1; }
    | variable_expression '=' assignment_expression         { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_ASSIGN, $1, $3); }
    | variable_expression "+=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_ADD_ASSIGN, $1, $3); }
    | variable_expression "-=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_SUB_ASSIGN, $1, $3); }
    | variable_expression "*=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_MUL_ASSIGN, $1, $3); }
    | variable_expression "/=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_DIV_ASSIGN, $1, $3); }
    | variable_expression "%=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_MOD_ASSIGN, $1, $3); }
    | variable_expression "&=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_AND_ASSIGN, $1, $3); }
    | variable_expression "^=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_XOR_ASSIGN, $1, $3); }
    | variable_expression "|=" assignment_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_OR_ASSIGN, $1, $3); }
    ;

variable_expression
    : "variable"                                            {   
                                                                auto var = $1;
                                                                const Type* _type = var->getType();
                                                                $$ = CreateNode<Expression>(ToLocation(@$), _type, std::move(var));
                                                            }
    | variable_expression '.' "identifier"                  {
                                                                auto var_expr = $1;
                                                                auto member_name = $3;
                                                                NodeT<Expression> result;
                                                                if(var_expr && var_expr->getSecond())
                                                                {
                                                                    const Type* _type = var_expr->getFirst()->getMemberType(driver, member_name->getValue());
                                                                    if(_type)
                                                                    {
                                                                        auto member = CreateNodeTyped<MemberVariable>(ToLocation(@$), std::move(var_expr->getSecond()), _type, member_name->getValue());
                                                                        result = CreateNodeTyped<Expression>(ToLocation(@$), _type, std::move(member));
                                                                    }
                                                                    else
                                                                        driver.error(ToLocation(@$), "Invalid member variable: " + member_name->getValue());
                                                                }
                                                                $$ = std::move(result);
                                                            }
    | variable_expression '[' expression ']'                {
                                                                auto                 var_expr = $1;
                                                                auto                 idx_expr = $3;
                                                                NodeT<Expression> result;
                                                                if(var_expr && var_expr->getSecond() && idx_expr && idx_expr->getFirst())
                                                                {
                                                                    if(idx_expr->getFirst()->getNodeName() != "int" && idx_expr->getFirst()->getNodeName() != "uint")
                                                                        driver.error(ToLocation(@3), "Invalid indexing type");
                                                                    else
                                                                    {
                                                                                                                                            
                                                                        const Type* _type = var_expr->getFirst()->getArrayElementType();
                                                                        if(_type)
                                                                        {
                                                                            auto arr = CreateNodeTyped<ArrayElementVariable>(ToLocation(@$), std::move(var_expr->getSecond()), _type, std::move(idx_expr->getSecond()));
                                                                            result = CreateNodeTyped<Expression>(ToLocation(@$), _type, std::move(arr));
                                                                        }
                                                                        else
                                                                            driver.error(ToLocation(@$), "The variable does not provide an array interface");
                                                                    }
                                                                }
                                                                $$ = std::move(result);
                                                            }
    ;

conditional_expression
    : logical_or_expression                                 { $$ = $1; }
    | logical_or_expression '?' conditional_expression ':' conditional_expression
                                                            {
                                                                auto                 cond_expr = $1;
                                                                auto                 true_expr = $3;
                                                                auto                 false_expr = $5;
                                                                NodeT<Expression>    result;
                                                                const Type*          _type = nullptr;
                                                                if(cond_expr && true_expr->getFirst() && false_expr && false_expr->getFirst())
                                                                {
                                                                    if(true_expr->getFirst()->hasImplicitConversionTo(false_expr->getFirst()))
                                                                        _type = false_expr->getFirst();
                                                                    else if(false_expr->getFirst()->hasImplicitConversionTo(false_expr->getFirst()))
                                                                        _type = true_expr->getFirst();
                                                                    if(_type)
                                                                        result = CreateNodeTyped<Expression>(ToLocation(@$), _type,
                                                                                                             CreateNode<TernaryIf>(ToLocation(@$), cond_expr ? std::move(cond_expr->getSecond()) : AST::Node(),
                                                                                                                                       true_expr ? std::move(true_expr->getSecond()) : AST::Node(),
                                                                                                                                       false_expr ? std::move(false_expr->getSecond()) : AST::Node()));
                                                                    else
                                                                        driver.error(ToLocation(@3), "Invalid implicit conversion between parameters of type " +
                                                                                         (true_expr ? true_expr->getFirst()->getNodeName() : "<unknown>") +
                                                                                         " and " + (false_expr ? false_expr->getFirst()->getNodeName() : "<unknown>"));
                                                                }
                                                                $$ = std::move(result);
                                                            }
    ;

logical_or_expression
    : logical_xor_expression                                { $$ = $1; }
    | logical_or_expression "||" logical_xor_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_OR, $1, $3); }
    ;

logical_xor_expression
    : logical_and_expression                                { $$ = $1; }
    | logical_xor_expression "^^" logical_and_expression    { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_XOR, $1, $3);  }
    ;

logical_and_expression
    : bitwise_or_expression                                 { $$ = $1; }
    | logical_and_expression "&&" bitwise_or_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_AND, $1, $3); }
    ;

bitwise_or_expression
    : bitwise_xor_expression                                { $$ = $1; }
    | bitwise_or_expression '|' bitwise_xor_expression      { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_OR, $1, $3);  }
    ;

bitwise_xor_expression
    : bitwise_and_expression                                { $$ = $1; }
    | bitwise_xor_expression '^' bitwise_and_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_XOR, $1, $3); }
    ;

bitwise_and_expression
    : equality_expression                                   { $$ = $1; }
    | bitwise_and_expression '&' equality_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_AND, $1, $3); }
    ;

equality_expression
    : relational_expression                                 { $$ = $1; }
    | equality_expression "==" relational_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_EQUAL, $1, $3); }
    | equality_expression "!=" relational_expression        { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_NEQUAL, $1, $3); }
    ;

relational_expression
    : bitwise_shift_expression                              { $$ = $1; }
    | relational_expression '<' bitwise_shift_expression    { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_LESS, $1, $3); }
    | relational_expression '>' bitwise_shift_expression    { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_GREATER, $1, $3); }
    | relational_expression "<=" bitwise_shift_expression   { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_LEQUAL, $1, $3); }
    | relational_expression ">=" bitwise_shift_expression   { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_GEQUAL, $1, $3); }
    ;

bitwise_shift_expression
    : additive_expression                                   { $$ = $1; }
    | bitwise_shift_expression ">>" additive_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_SHIFT_RIGHT, $1, $3); }
    | bitwise_shift_expression "<<" additive_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_BITWISE_SHIFT_LEFT, $1, $3); }
    ;

additive_expression
    : multiplicative_expression                             { $$ = $1; }
    | additive_expression '+' multiplicative_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_ADD, $1, $3); }
    | additive_expression '-' multiplicative_expression     { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_SUBTRACT, $1, $3); }
    ;

multiplicative_expression
    : prefix_expression                                     { $$ = $1; }
    | multiplicative_expression '*' prefix_expression       { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_MULTIPLY, $1, $3); }
    | multiplicative_expression '/' prefix_expression       { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_DIVIDE, $1, $3); }
    | multiplicative_expression '%' prefix_expression       { $$ = CreateBinaryOperator(driver, ToLocation(@$), TGE_EFFECT_MODULUS, $1, $3); }
    ;

prefix_expression
    : suffix_expression                                     { $$ = $1; }
    | '-' prefix_expression                                 { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_NEGATE, $2); }
    | '+' prefix_expression                                 { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_POSITIVE, $2); }
    | '!' prefix_expression                                 { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_NOT, $2); }
    | '~' prefix_expression                                 { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_COMPLEMENT, $2); }
    | "++" prefix_expression                                { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_PRE_INCR, $2); }
    | "--" prefix_expression                                { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_PRE_DECR, $2); }
    ;

suffix_expression
    : parentheses_expression                                { $$ = $1; }
    | "type" '(' function_arg_list ')'                      {
                                                                auto _type = $1;
                                                                auto arg_list = $3;
                                                                NodeT<Expression> result;
                                                                TGE_ASSERT(_type, "Expecting valid type node");
                                                                if(_type->hasValidConstructor(arg_list.get()))
                                                                {
                                                                    auto constructor_call = _type->createConstructorCall(ToLocation(@$), std::move(arg_list));
                                                                    result = CreateNodeTyped<Expression>(ToLocation(@$), _type.get(), std::move(constructor_call));
                                                                }
                                                                else
                                                                    driver.error(ToLocation(@$), "Invalid constructor call");
                                                                $$ = std::move(result);
                                                            }
    | function_call                                         { $$ = $1; }
    | suffix_expression "++"                                { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_POST_INCR, $1); }
    | suffix_expression "--"                                { $$ = CreateUnaryOperator(driver, ToLocation(@$), TGE_EFFECT_POST_DECR, $1); }
    | suffix_expression '.' "identifier"                    {
                                                                auto suffix_expr = $1;
                                                                auto idx_expr = $3; 
                                                                NodeT<Expression> result;
                                                                if(suffix_expr && suffix_expr->getFirst() && idx_expr)
                                                                {
                                                                    const Type* t = suffix_expr->getFirst()->getMemberType(driver, idx_expr->getValue());
                                                                    result = CreateNodeTyped<Expression>(ToLocation(@$), t, t ? CreateNode<MemberVariable>(ToLocation(@$), std::move(suffix_expr->getSecond()), t, idx_expr->getValue()) : AST::Node());
                                                                }
                                                                $$ = std::move(result);
                                                            }
    | suffix_expression  '[' expression ']'                 {
                                                                auto suffix_expr = $1;
                                                                auto idx_expr = $3;   
                                                                NodeT<Expression> result;
                                                                if(suffix_expr && suffix_expr->getFirst() && idx_expr && idx_expr->getFirst())
                                                                {
                                                                    if(idx_expr->getFirst()->getNodeName() != "int" && idx_expr->getFirst()->getNodeName() != "uint")
                                                                        driver.error(ToLocation(@3), "Invalid indexing type");
                                                                    else
                                                                    {
                                                                        const Type* t = suffix_expr->getFirst()->getArrayElementType();
                                                                        result = CreateNodeTyped<Expression>(ToLocation(@$), t, t ? CreateNode<ArrayElementVariable>(ToLocation(@$), std::move(suffix_expr->getSecond()), t, std::move(idx_expr->getSecond())) : AST::Node());
                                                                    }
                                                                }
                                                                $$ = std::move(result);
                                                            }
    ;

function_call
    : "function" '(' function_arg_list ')'                  {
                                                                NodeT<Expression> result;
                                                                auto              function_set = $1;
                                                                auto              arg_list = $3;
                                                                if(arg_list)
                                                                {
                                                                    auto* func = function_set->getFunction(arg_list.get());
                                                                    if(func)
                                                                    {
                                                                        auto func_call = func->createFunctionCall(ToLocation(@$), std::move(arg_list));
                                                                        result = CreateNodeTyped<Expression>(ToLocation(@$), func_call->getFunction()->getReturnType(), std::move(func_call));
                                                                    }
                                                                    else
                                                                    {
                                                                        std::stringstream ss;
                                                                        Printer func_printer(ss, 0);
                                                                        ss << "Invalid function call to undeclared function: " << function_set->getNodeName() << "(";
                                                                        for(List::iterator i = arg_list->current(), iend = arg_list->end(); i != iend; ++i)
                                                                        {
                                                                            if(i != arg_list->current())
                                                                                ss << ", ";
                                                                            auto ptr = i->extract<Expression>();
                                                                            if(ptr && i && ptr->getFirst())
                                                                                ss << ptr->getFirst()->getNodeName();
                                                                            else
                                                                                ss << "<unknown>";
                                                                        }
                                                                        ss << ")\n"
                                                                              "Candidates are:\n";
                                                                        for(size_t i = 0, iend = function_set->getFunctionCount(); i != iend; ++i)
                                                                            ss << "\t", func_printer.visit(function_set->getFunction(i)), ss << '\n';
                                                                        driver.error(ToLocation(@$), ss.str());
                                                                    }
                                                                }
                                                                $$ = std::move(result);
                                                            }
    ;

parentheses_expression
    : variable_expression                                   { $$ = $1; }
    | scalar_expression                                     { $$ = $1; }
    | '(' expression ')'                                    {
                                                                auto expr = $2;
                                                                $$ = CreateNode<Expression>(ToLocation(@$), expr ? expr->getFirst() : nullptr,
                                                                                            CreateNode<Parentheses>(ToLocation(@$), expr ? std::move(expr->getSecond()) : AST::Node()));
                                                            }
    ;

scalar_expression
    : "integer"                                             { const Type* rt = driver.find("int"); $$ = CreateNodeTyped<Expression>(ToLocation(@$), rt, $1); }
    | "unsigned integer"                                    { const Type* rt = driver.find("uint"); $$ = CreateNodeTyped<Expression>(ToLocation(@$), rt, $1); }
    | "float"                                               { const Type* rt = driver.find("float"); $$ = CreateNodeTyped<Expression>(ToLocation(@$), rt, $1); }
    | "boolean"                                             { const Type* rt = driver.find("bool"); $$ = CreateNodeTyped<Expression>(ToLocation(@$), rt, $1);}
    ;

function_arg_list
    : /* empty */                                           { $$ = NodeT<List>(); }
    | conditional_expression                                { $$ = CreateNodeTyped<ListElement>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, $1, NodeT<List>()); }
    | conditional_expression ',' function_arg_list          { $$ = CreateNodeTyped<ListElement>(ToLocation(@$), TGE_AST_COMMA_SEPARATED_LIST, $1, $3); }
    ;

%%

namespace Tempest
{
namespace Shader
{
void Parser::error(const Parser::location_type& l, const std::string& m)
{
    driver.error(ToLocation(l), m);
}

NodeT<Expression> CreateBinaryOperator(Driver& driver, Location loc, BinaryOperatorType binop_type,
                                          NodeT<Expression> left_expr, NodeT<Expression> right_expr)
{
    if(left_expr && right_expr)
    {
        auto* left_type = left_expr->getFirst();
        const Type* _type = left_type->binaryOperatorResultType(driver, binop_type, right_expr->getFirst());
        if(_type)
            return CreateNode<Expression>(loc, _type, CreateNode<BinaryOperator>(loc, binop_type, std::move(left_expr->getSecond()), std::move(right_expr->getSecond())));
        else
            driver.error(loc, "Invalid binary operation between expressions of type " +
                              left_expr->getFirst()->getNodeName() + " and " + right_expr->getFirst()->getNodeName());
    }
    return NodeT<Expression>();
}

NodeT<Expression> CreateUnaryOperator(Driver& driver, Location loc, UnaryOperatorType uniop_type,
                                         NodeT<Expression> expr)
{
    if(expr)
    {
        const Type* _type = expr->getFirst()->unaryOperatorResultType(driver, uniop_type);
        if(_type)
            return CreateNodeTyped<Expression>(loc, _type, CreateNode<UnaryOperator>(loc, uniop_type, std::move(expr->getSecond())));
        else
            driver.error(loc, "Invalid unary operation applied on expression of type " + expr->getFirst()->getNodeName());
    }
    return NodeT<Expression>();
}

void ErrorRedefinition(Driver& driver, Location loc, NodeT<VariableRef> var)
{
    TGE_ASSERT(var, "Expected at least valid variable");
    std::stringstream err;
    auto orig_loc = var.getDeclarationLocation();
    if(loc.filename)
        err << "Redefinition of variable \"" << var->getNodeName() << "\". Previously declared at: " << orig_loc;
    else
        err << "Redefinition of standard variable \"" << var->getNodeName() << "\".";
    driver.error(loc, err.str());
}
}
}
