/*   The MIT License
*
*   Tempest Engine
*   Copyright (c) 2015 Zdravko Velinov
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

#include "tempest/shader/shader-parser.hh"
#include "tempest/shader/shader-driver.hh"
#include "tempest/shader/shader-ast.hh"

#define UNHANDLED false
#define HANDLED true

YY_DECL;

namespace Tempest
{
namespace Shader
{
std::ostream& operator<<(std::ostream& os, ShaderToken token)
{
    const char* name;

    if((uint32)token > (uint32)ShaderToken::Count)
    {
        char c = '!' + (uint32)token - (uint32)ShaderToken::Count;
        os << c;
        return os;
    }

    switch(token)
    {
#define SHADER_TOKEN(token_enum, token_name) case ShaderToken::token_enum: name = token_name; break;
#include "tempest/shader/shader-tokens.hh"
#undef SHADER_TOKEN
default: TGE_ASSERT(false, "Unknown token"); os << "unknown"; return os;
    }

    os << name;
    return os;
}

bool IsCharacterToken(ShaderToken token, char _c)
{
    return token == ToCharacterToken(_c);
}

Parser::Parser(Shader::Driver& driver)
    :   m_Driver(driver)
{
    m_CurrentLocation.filename = &driver.__FileName;
}

void Parser::parseToken()
{
    if(m_Reprocess)
    {
        m_Reprocess = false;
        return;
    }

    m_CurrentToken = ShaderLexer(&m_CurrentNode, &m_CurrentLocation, m_Driver);
}


AST::Node Parser::statementList()
{
    auto top_element = m_NodeStack.size();

    for(;;)
    {
        auto status = basicVariableDeclaration()
                   //|| selectionStatement()
                   //|| switchStatement()
                   //|| iterationStatement()
                   || expressionStatement()
                   //|| blockStatement()
                   //|| jumpStatement()
                   //|| optionStatement()
                   ;
        if(!status)
            break;
    }

    return collapseStackToList(ListType::SemicolonSeparated, top_element);
}

bool Parser::expressionStatement()
{
    auto expr_loc = m_CurrentLocation;
    auto expr = expression();
    if(!expr)
    {
        return UNHANDLED;
    }
    auto status = expect(ToCharacterToken(';'));
    if(!status)
    {
        m_Driver.error(expr_loc, "Invalid statement");
        skipToToken(';');
        return HANDLED;
    }
    m_NodeStack.emplace_back(std::move(expr->getSecond()));
    return HANDLED;
}

BinaryOperatorType ToAssignmentOperator(ShaderToken token)
{
    switch(token)
    {
    default: break;
    case ToCharacterToken('='): return BinaryOperatorType::Assign;
    case ShaderToken::AddAssign: return BinaryOperatorType::AddAssign;
    case ShaderToken::SubAssign: return BinaryOperatorType::SubtractAssign;
    case ShaderToken::MulAssign: return BinaryOperatorType::MultiplyAssign;
    case ShaderToken::DivAssign: return BinaryOperatorType::DivideAssign;
    case ShaderToken::ModAssign: return BinaryOperatorType::ModulusAssign;
    case ShaderToken::BitwiseAndAssign: return BinaryOperatorType::BitwiseAndAssign;
    case ShaderToken::BitwiseXorAssign: return BinaryOperatorType::BitwiseXorAssign;
    case ShaderToken::BitwiseOrAssign: return BinaryOperatorType::BitwiseOrAssign;
    }

    return BinaryOperatorType::Unknown;
}

AST::NodeT<Expression> Parser::assignmentExpression()
{
    auto expr_loc = m_CurrentLocation;
    auto expr = conditionalExpression();
    if(!expr)
    {
        return AST::NodeT<Expression>();
    }

    struct AssignmentOperation
    {
        Location location;
        BinaryOperatorType operation;
    };

    std::vector<AssignmentOperation> operations;
    std::vector<AST::NodeT<Expression>> exprs;
    exprs.push_back(std::move(expr));

    for(;;)
    {
        auto expr_loc = m_CurrentLocation;
        parseToken();

        auto oper = ToAssignmentOperator(m_CurrentToken);

        if(oper == BinaryOperatorType::Unknown)
        {
            m_Reprocess = true;
            break;
        }

        parseToken();
        auto expr = conditionalExpression();
        if(!expr)
        {
            m_Driver.error(expr_loc, "Invalid assignment expression");
            return AST::NodeT<Expression>();
        }

        operations.emplace_back(AssignmentOperation{ expr_loc, oper });
        exprs.emplace_back(std::move(expr));
    }
    
    auto rhs_expr = std::move(exprs.back());

    for(size_t i = exprs.size() - 1; i-- > 0;)
    {
        auto& oper = operations[i];
        auto& lhs_expr = exprs[i];

        auto* _type = lhs_expr->getFirst()->binaryOperatorResultType(m_Driver, oper.operation, rhs_expr->getFirst());
        rhs_expr = CreateNodeTyped<Expression>(oper.location, _type, CreateNode<BinaryOperator>(oper.location, oper.operation, std::move(lhs_expr->getSecond()), std::move(rhs_expr->getSecond())));
    }

    return std::move(rhs_expr);
}

AST::NodeT<Expression> Parser::expression()
{
    // TODO: Comma expression
    return assignmentExpression();
}

bool IsUnaryPrefixOperator(ShaderToken token)
{
    switch(token)
    {
    default: break;
    case ToCharacterToken('-'):
    case ToCharacterToken('+'):
    case ToCharacterToken('!'):
    case ToCharacterToken('~'):
    case ShaderToken::Increment:
    case ShaderToken::Decrement:
        return true;
    }
    return false;
}

bool IsUnarySuffixOperator(ShaderToken token)
{
    switch(token)
    {
    default: break;
    case ShaderToken::Increment:
    case ShaderToken::Decrement:
        return true;
    }
    return false;
}

UnaryOperatorType ToUnaryOperator(ShaderToken token, bool is_pre)
{
    switch(token)
    {
    default: break;
    case ToCharacterToken('-'): return UnaryOperatorType::Negative;
    case ToCharacterToken('+'): return UnaryOperatorType::Positive;
    case ToCharacterToken('!'): return UnaryOperatorType::Negative;
    case ToCharacterToken('~'): return UnaryOperatorType::Complement;
    case ShaderToken::Increment: return is_pre ? UnaryOperatorType::PreIncrement : UnaryOperatorType::PostIncrement;
    case ShaderToken::Decrement: return is_pre ? UnaryOperatorType::PreDecrement : UnaryOperatorType::PostDecrement;
    }
    return UnaryOperatorType::Unknown;
}

AST::NodeT<List> Parser::functionArgList()
{
    auto loc = m_CurrentLocation;
    auto status = expect(ToCharacterToken('('));
    if(!status)
    {
        m_Driver.error(loc, "Invalid function argument list");
        return AST::NodeT<List>();
    }

    auto top_element = m_NodeStack.size();

    parseToken();
    if(m_CurrentToken == ToCharacterToken(')'))
        return AST::NodeT<List>();

    for(;;)
    {
        auto expr = conditionalExpression();
        bool status = expr;
        if(status)
        {
            parseToken();
            if(IsCharacterToken(m_CurrentToken, ')'))
            {
                m_NodeStack.push_back(std::move(expr));
                break;
            }
            else if(!IsCharacterToken(m_CurrentToken, ','))
            {
                status = false;
            }
        }

        if(!status)
        {
            m_Driver.error(loc, "Invalid function argument list");
            skipToToken(')');
            return AST::NodeT<List>();
        }

        m_NodeStack.push_back(std::move(expr));

        parseToken();
    }

    return collapseStackToList(ListType::CommaSeparated, top_element);
}

AST::NodeT<Expression> Parser::paranthesesExpression()
{
    auto par_loc = m_CurrentLocation;
    switch(m_CurrentToken)
    {
    case ShaderToken::Function:
    {
        parseToken();
        auto arg_list = functionArgList();

        auto function_set = m_CurrentNode.extract<FunctionSet>();
        auto* func = function_set->getFunction(arg_list);
        if(!func)
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
            m_Driver.error(par_loc, ss.str());
        }

        auto func_call = func->createFunctionCall(par_loc, std::move(arg_list));
        return CreateNodeTyped<Expression>(par_loc, func_call->getFunction()->getReturnType(), std::move(func_call));
    } break;
    case ShaderToken::Type:
    {
        auto _type = m_CurrentNode.extract<Type>();
        auto arg_list = functionArgList();
        if(!arg_list)
        {
            m_Driver.error(par_loc, "Invalid empty constructor call");
            return NodeT<Expression>();
        }

        TGE_ASSERT(_type, "Expecting valid type node");
        if(!_type->hasValidConstructor(arg_list.get()))
        {
            m_Driver.error(par_loc, "Invalid constructor call");
            return NodeT<Expression>();
        }
        auto constructor_call = _type->createConstructorCall(par_loc, std::move(arg_list));
        return CreateNodeTyped<Expression>(par_loc, _type, std::move(constructor_call));
    } break;
    case ShaderToken::Variable:
    {
        auto* var = m_CurrentNode.extract<Variable>();
        return CreateNode<Expression>(par_loc, var->getType(), std::move(m_CurrentNode));
    } break;
    case ToCharacterToken('('):
    {
        auto expr = expression();
        bool status = expr;
        if(status)
        {
            parseToken();
            auto status = expect(ToCharacterToken(')'));
        }

        if(!status)
        {
            m_Driver.error(par_loc, "Invalid parantheses expression");
            skipToToken(')');
            return AST::NodeT<Expression>();
        }
        return CreateNode<Expression>(par_loc, expr ? expr->getFirst() : nullptr,
                                      CreateNode<Parentheses>(par_loc, expr ? std::move(expr->getSecond()) : AST::Node()));
    } break;
    case ShaderToken::Integer:
    {
        const Type* rt = m_Driver.find("int");
        return CreateNodeTyped<Expression>(par_loc, rt, std::move(m_CurrentNode));
    } break;
    case ShaderToken::Unsigned:
    {
        const Type* rt = m_Driver.find("uint");
        return CreateNodeTyped<Expression>(par_loc, rt, std::move(m_CurrentNode));
    } break;
    case ShaderToken::Float:
    {
        const Type* rt = m_Driver.find("float");
        return CreateNodeTyped<Expression>(par_loc, rt, std::move(m_CurrentNode));
    } break;
    case ShaderToken::Boolean:
    {
        const Type* rt = m_Driver.find("bool"); 
        return CreateNodeTyped<Expression>(par_loc, rt, std::move(m_CurrentNode));
    } break;
    default: break;
    }
    return AST::NodeT<Expression>();
}

// BUG?: Conflict between declaration and constructor call
AST::NodeT<Expression> Parser::suffixExpression()
{
    auto suf_loc = m_CurrentLocation;
    auto expr = paranthesesExpression();
    if(!expr)
    {
        return AST::NodeT<Expression>();
    }

    for(;;)
    {
        parseToken();
        switch(m_CurrentToken)
        {
        default: m_Reprocess = true; goto break_loop;
        case ShaderToken::Increment:
        {
            expr = CreateNode<UnaryOperator>(m_CurrentLocation, UnaryOperatorType::PostIncrement, std::move(expr));
        } break;
        case ShaderToken::Decrement:
        {
            expr = CreateNode<UnaryOperator>(m_CurrentLocation, UnaryOperatorType::PostDecrement, std::move(expr));
        } break;
        case ToCharacterToken('.'):
        {
            auto mem_expr = m_CurrentLocation;
            auto member = expectNode(ShaderToken::Identifier);
            if(!member)
            {
                return AST::NodeT<Expression>();
            }
            auto value = member.extract<Identifier>()->getValue();
            const Type* t = expr->getFirst()->getMemberType(m_Driver, value);
            expr = CreateNodeTyped<Expression>(mem_expr, t, t ? CreateNode<MemberVariable>(mem_expr, std::move(expr->getSecond()), t, value) : AST::Node());
        } break;
        case ToCharacterToken('['):
        {
            parseToken();
            auto idx_loc = m_CurrentLocation;
            auto idx_expr = expression();
            if(!idx_expr || !expr->getFirst() || !idx_expr->getFirst())
            {
                return AST::NodeT<Expression>();
            }
            auto status = expect(ToCharacterToken(']'));
            if(!status)
            {
                m_Driver.error(idx_loc, "Invalid indexing expression");
                skipToToken(']');
                return AST::NodeT<Expression>();
            }

            if(idx_expr->getFirst()->getNodeName() != "int" && idx_expr->getFirst()->getNodeName() != "uint")
            {
                m_Driver.error(idx_loc, "Invalid indexing type");
            }
            else
            {
                const Type* t = expr->getFirst()->getArrayElementType();
                expr = CreateNodeTyped<Expression>(suf_loc, t, t ? CreateNode<ArrayElementVariable>(suf_loc, std::move(expr->getSecond()), t, std::move(idx_expr->getSecond())) : AST::Node());
            }
        } break;
        }
    }
break_loop:;

    return std::move(expr);
}

AST::NodeT<Expression> Parser::prefixExpression()
{
    if(IsUnaryPrefixOperator(m_CurrentToken))
    {
        auto expr_loc = m_CurrentLocation;
        parseToken();
        struct UnaryOperation
        {
            Location            location;
            UnaryOperatorType   operation;
        };

        std::vector<UnaryOperation> unary_operators;

        while(IsUnaryPrefixOperator(m_CurrentToken))
        {
            unary_operators.emplace_back(UnaryOperation{ m_CurrentLocation, ToUnaryOperator(m_CurrentToken, true) });
            parseToken();
        }

        auto expr = suffixExpression();
        if(!expr)
        {
            m_Driver.error(expr_loc, "Invalid prefix unary operation");
            return AST::NodeT<Expression>();
        }

        auto result_node = std::move(expr->getSecond());
        const Type* _type = expr->getFirst();
        for(size_t i = unary_operators.size(); i-- > 0;)
        {
            auto& op = unary_operators[i];
            _type = expr->getFirst()->unaryOperatorResultType(m_Driver, op.operation);
            result_node = CreateNode<UnaryOperator>(op.location, op.operation, std::move(result_node));
        }
        return CreateNode<Expression>(expr_loc, _type, std::move(result_node));
    }
    else
    {
        return suffixExpression();
    }
}

bool IsBinaryOperator(ShaderToken op)
{
    switch(op)
    {
    default: break;
    case ToCharacterToken('*'):
    case ToCharacterToken('/'):
    case ToCharacterToken('%'):
    case ToCharacterToken('+'):
    case ToCharacterToken('-'):
    case ShaderToken::ShiftLeft:
    case ShaderToken::ShiftRight:
    case ToCharacterToken('<'):
    case ToCharacterToken('>'):
    case ShaderToken::GreaterEqual:
    case ShaderToken::LessEqual:
    case ShaderToken::Equal:
    case ShaderToken::NotEqual:
    case ToCharacterToken('&'):
    case ToCharacterToken('^'):
    case ToCharacterToken('|'):
    case ShaderToken::LogicalAnd:
    case ShaderToken::LogicalOr:
    case ShaderToken::LogicalXor:
        return true;
    }
    return false;
}

void BinaryOperatorInfo(ShaderToken op, BinaryOperatorType* _type, size_t* precedence)
{
    switch(op)
    {
    default: TGE_ASSERT(false, "Unknown binary operator"); *_type = BinaryOperatorType::Unknown, *precedence = 0; break;
    case ToCharacterToken('*'): *_type = BinaryOperatorType::Multiply; *precedence = 3; break;
    case ToCharacterToken('/'): *_type = BinaryOperatorType::Divide; *precedence = 3; break;
    case ToCharacterToken('%'): *_type = BinaryOperatorType::Modulus; *precedence = 3; break;
    case ToCharacterToken('+'): *_type = BinaryOperatorType::Add; *precedence = 4; break;
    case ToCharacterToken('-'): *_type = BinaryOperatorType::Subtract; *precedence = 4; break;
    case ShaderToken::ShiftLeft: *_type = BinaryOperatorType::BitwiseShiftLeft; *precedence = 5; break;
    case ShaderToken::ShiftRight: *_type = BinaryOperatorType::BitwiseShiftRight; *precedence = 5; break;
    case ToCharacterToken('<'): *_type = BinaryOperatorType::Less; *precedence = 6; break;
    case ToCharacterToken('>'): *_type = BinaryOperatorType::Greater; *precedence = 6; break;
    case ShaderToken::LessEqual: *_type = BinaryOperatorType::LessEqual; *precedence = 6; break;
    case ShaderToken::GreaterEqual: *_type = BinaryOperatorType::GreaterEqual; *precedence = 6; break;
    case ShaderToken::Equal: *_type = BinaryOperatorType::Equal; *precedence = 7; break;
    case ShaderToken::NotEqual: *_type = BinaryOperatorType::NotEqual; *precedence = 7; break;
    case ToCharacterToken('&'): *_type = BinaryOperatorType::BitwiseAnd; *precedence = 8; break;
    case ToCharacterToken('^'): *_type = BinaryOperatorType::BitwiseXor; *precedence = 9; break;
    case ToCharacterToken('|'): *_type = BinaryOperatorType::BitwiseOr; *precedence = 10; break;
    case ShaderToken::LogicalAnd: *_type = BinaryOperatorType::And; *precedence = 11; break;
    case ShaderToken::LogicalOr: *_type = BinaryOperatorType::Or; *precedence = 12; break;
    case ShaderToken::LogicalXor: *_type = BinaryOperatorType::Xor; *precedence = 12; break;
    }
}

AST::NodeT<Expression> Parser::binaryExpression()
{
    auto loc = m_CurrentLocation;
    auto expr = prefixExpression();
    if(!expr)
    {
        return AST::NodeT<Expression>();
    }

    parseToken();
    if(!IsBinaryOperator(m_CurrentToken))
    {
        m_Reprocess = true;
        return expr;
    }

    BinaryOperatorType _op, last_op;
    size_t precedence;
    BinaryOperatorInfo(m_CurrentToken, &_op, &precedence);
    last_op = _op;
    size_t last_prec = precedence;

    struct OperationStart
    {
        size_t             start;
        size_t             precedence;
        BinaryOperatorType operation;
    };

    std::vector<OperationStart> oper_stack;
    std::vector<NodeT<Expression>> expr_stack;
    expr_stack.push_back(std::move(expr));

    parseToken();
    for(;;)
    {
        auto rhs_loc = m_CurrentLocation;
        auto rhs_expr = prefixExpression();

        if(!rhs_expr)
        {
            m_Driver.error(rhs_loc, "Invalid right-hand side operand");
        }

        // Collapse the operator stack
        if(precedence > last_prec)
        {
            TGE_ASSERT(oper_stack.size() - oper_stack.back().start >= 2, "Stack does not contain at least two expression to create binary operation");
            auto rhs_expr = std::move(expr_stack.back());
            expr_stack.pop_back();
            while(!expr_stack.empty() && precedence > last_prec)
            {
                auto lhs_expr = std::move(expr_stack.back());
                const Type* _type = lhs_expr->getFirst()->binaryOperatorResultType(m_Driver, last_op, rhs_expr->getFirst());
                auto binop_loc = lhs_expr.getDeclarationLocation();
                rhs_expr = CreateNodeTyped<Expression>(binop_loc, _type, CreateNode<BinaryOperator>(binop_loc, last_op, std::move(lhs_expr), std::move(rhs_expr)));
                expr_stack.pop_back();
                if(!oper_stack.empty())
                {
                    auto& oper = oper_stack.back();
                    if(oper.start == expr_stack.size())
                    {
                        last_prec = oper.precedence;
                        last_op = oper.operation;
                        oper_stack.pop_back();
                    }
                }
            }
            expr_stack.push_back(std::move(rhs_expr));
        }

        if(last_prec < precedence)
        {
            oper_stack.push_back(OperationStart{ expr_stack.size() - 1, precedence, _op });
        }

        expr_stack.push_back(std::move(rhs_expr));

        parseToken();
        if(!IsBinaryOperator(m_CurrentToken))
        {
            m_Reprocess = true;
            break;
        }

        last_prec = precedence;
        last_op = _op;
        BinaryOperatorInfo(m_CurrentToken, &_op, &precedence);
        loc = rhs_loc;
    }

    TGE_ASSERT(!expr_stack.empty(), "Broken stack");
    auto rhs_expr = std::move(expr_stack.back());
    expr_stack.pop_back();
    while(!expr_stack.empty())
    {
        auto lhs_expr = std::move(expr_stack.back());
        const Type* _type = lhs_expr->getFirst()->binaryOperatorResultType(m_Driver, last_op, rhs_expr->getFirst());
        auto binop_loc = lhs_expr.getDeclarationLocation();
        rhs_expr = CreateNodeTyped<Expression>(binop_loc, _type, CreateNode<BinaryOperator>(binop_loc, last_op, std::move(lhs_expr->getSecond()), std::move(rhs_expr->getSecond())));
        expr_stack.pop_back();
        if(!oper_stack.empty())
        {
            auto& oper = oper_stack.back();
            if(oper.start == expr_stack.size())
            {
                last_op = oper.operation;
                oper_stack.pop_back();
            }
        }
    }
    return std::move(rhs_expr);
}

AST::NodeT<Expression> Parser::conditionalExpression()
{
    auto expr_loc = m_CurrentLocation;
    auto cond_expr = binaryExpression();
    bool status = cond_expr;
    if(!status)
        return AST::Node();
    parseToken();
    if(!IsCharacterToken(m_CurrentToken, '?'))
    {
        m_Reprocess = true;
        return std::move(cond_expr);
    }

    parseToken();

    auto result_loc = m_CurrentLocation;
    auto true_expr = conditionalExpression();
    NodeT<Expression> false_expr;
    status = true_expr;
    if(status)
    {
        status = expect(ToCharacterToken(':'));
        if(status)
        {
            false_expr = conditionalExpression();
        }
    }

    if(!status)
    {
        m_Driver.error(expr_loc, "Invalid expression");
        return AST::Node();
    }

    auto stack_size = m_NodeStack.size();
    TGE_ASSERT(stack_size >= 3, "Broken stack");

    NodeT<Expression>    result;
    const Type*          _type = nullptr;
    if(cond_expr && true_expr->getFirst() && false_expr && false_expr->getFirst())
    {
        if(true_expr->getFirst()->hasImplicitConversionTo(false_expr->getFirst()))
        {
            _type = false_expr->getFirst();
        }
        else if(false_expr->getFirst()->hasImplicitConversionTo(false_expr->getFirst()))
        {
            _type = true_expr->getFirst();
        }

        if(_type)
        {
            result = CreateNodeTyped<Expression>(expr_loc, _type,
                                                 CreateNode<TernaryIf>(expr_loc, cond_expr ? std::move(cond_expr->getSecond()) : AST::Node(),
                                                 true_expr ? std::move(true_expr->getSecond()) : AST::Node(),
                                                 false_expr ? std::move(false_expr->getSecond()) : AST::Node()));
        }
        else
        {
            m_Driver.error(result_loc, "Invalid implicit conversion between parameters of type " +
                                       (true_expr ? true_expr->getFirst()->getNodeName() : "<unknown>") +
                                       " and " + (false_expr ? false_expr->getFirst()->getNodeName() : "<unknown>"));
        }
    }
    return std::move(result);
}

bool Parser::basicVariableDeclaration()
{
    auto var_loc = m_CurrentLocation;
    auto decl_loc = m_CurrentLocation;
    AST::Node var_node = variable();
    if(!var_node)
        return UNHANDLED;

    AST::Node decl_list;

    auto top_element = m_NodeStack.size();

    parseToken();

    for(;;)
    {
        auto* var = var_node.extract<Variable>();
        
        if(IsCharacterToken(m_CurrentToken, '='))
        {
            parseToken();
            auto cond_loc = m_CurrentLocation;
            auto cond_node = conditionalExpression();
            if(!cond_node)
            {
                m_Driver.error(cond_loc, "Invalid assignment declaration");
            }
            parseToken();

            var_node = CreateNode<BinaryOperator>(var_loc, BinaryOperatorType::Assign, std::move(var_node), std::move(cond_node));
        }
        
        m_NodeStack.emplace_back(std::move(var_node));

        if(IsCharacterToken(m_CurrentToken, ';'))
           break;

        if(!IsCharacterToken(m_CurrentToken, ','))
        {
            m_Reprocess = true;
            m_Driver.error(decl_loc, "Invalid declaration");
            m_NodeStack.emplace_back(AST::Node());
            return HANDLED;
        }

        var_loc = m_CurrentLocation;
        AST::Node var_node = variable();
    }

    if(m_NodeStack.size() - top_element == 1)
    {
        decl_list = std::move(m_NodeStack.back());
        m_NodeStack.pop_back();
    }
    else
    {
        decl_list = collapseStackToList(ListType::CommaSeparated, top_element);
    }

    m_NodeStack.emplace_back(CreateNode<Declaration>(decl_loc, std::move(decl_list)));

    return HANDLED;
}

AST::Node Parser::variable()
{
    if(m_CurrentToken != ShaderToken::Type)
        return AST::Node();

    auto* _type = m_CurrentNode.extract<Shader::Type>();

    auto var_loc = m_CurrentLocation;
    AST::Node ident = expectRedefCheck(ShaderToken::Identifier, ShaderToken::Variable);
    if(!ident)
        return AST::Node();
    parseToken();
    if(IsCharacterToken(m_CurrentToken, '['))
    {
        TGE_ASSERT(false, "Stub");
        AST::Node expr = expression();
        if(expr)
        {
            parseToken();
            expr = std::move(m_NodeStack.back());
            m_NodeStack.pop_back();
        }
        auto result = expect(ToCharacterToken(']'));
        if(!result)
        {
            m_Driver.error(var_loc, "Invalid array declaration");
            skipToToken(']');
            return AST::Node();
        }

        auto arr_type = m_Driver.createInternalType<ArrayType>(var_loc, _type, expr ? std::move(expr.extract<Shader::Expression>()->getSecond()) : AST::Node());
        return m_Driver.createStackNode<Variable>(var_loc, arr_type.get(), ident.extract<Identifier>()->getValue());
    }
    else
    {
        m_Reprocess = true;
        return m_Driver.createStackNode<Variable>(var_loc, _type, ident.extract<Shader::Identifier>()->getValue());
    }
}

bool Parser::function()
{
    // Basically, we can't declare regular variables in global space, so it is definitely a function
    if(m_CurrentToken != ShaderToken::Type && m_CurrentToken != ShaderToken::VoidType)
        return HANDLED;

    auto* return_type = m_CurrentNode.extract<Shader::Type>();

    auto func_decl_loc = m_CurrentLocation;

    parseToken();
    auto ident_type = m_CurrentToken;
    if(ident_type != ShaderToken::Identifier &&
       ident_type != ShaderToken::Function)
    {
        m_NodeStack.emplace_back(AST::Node());
        m_Driver.error(func_decl_loc, "Invalid declaration. Might be a function declaration, but it should be type followed by identifier.");
        skipDeclarationOrDefinition();
        return HANDLED;
    }
    auto ident = std::move(m_CurrentNode);
    bool status = ident;
    AST::Node args;

    {
        m_Driver.beginBlock();
        auto& driver = m_Driver;
        auto at_exit = CreateAtScopeExit([&driver]() { driver.endBlock(); });

        if(status)
        {
            status = expect(ToCharacterToken('('));
            if(!status)
            {
                skipDeclarationOrDefinition();
                return HANDLED;
            }

            auto top_element = m_NodeStack.size();

            parseToken();

            if(!IsCharacterToken(m_CurrentToken, ')'))
            {
                for(;;)
                {
                    StorageQualifier storage_mode;
                    auto var_loc = m_CurrentLocation;
                    switch(m_CurrentToken)
                    {
                    case ShaderToken::InQualifier:
                    {
                        storage_mode = StorageQualifier::In;
                        parseToken();
                    } break;
                    case ShaderToken::OutQualifier:
                    {
                        storage_mode = StorageQualifier::Out;
                        parseToken();
                    } break;
                    case ShaderToken::InOutQualifier:
                    {
                        storage_mode = StorageQualifier::InOut;
                        parseToken();
                    } break;
                    }

                    auto var = variable();
                    if(!var)
                    {
                        status = false;
                        m_Driver.error(var_loc, "Invalid function argument declaration");
                        break;
                    }

                    m_NodeStack.emplace_back(CreateNode<Declaration>(var_loc, std::move(var)));

                    if(IsCharacterToken(m_CurrentToken, ')'))
                        break;

                    status = expect(ToCharacterToken(','));
                }

                args = collapseStackToList(ListType::SemicolonSeparated, top_element);
            }
        }

        if(!status)
        {
            skipToToken(')');
            parseToken();
            skipDeclarationOrDefinition();
            m_NodeStack.emplace_back(AST::Node());
            return HANDLED;
        }
    }

    parseToken();
    if(IsCharacterToken(m_CurrentToken, ';'))
    {
        NodeT<Reference<FunctionDeclaration>> func;
        FunctionDeclaration* func_decl_ptr = nullptr;
        if(ident)
        {
            if(ident_type == ShaderToken::Identifier)
            {
                string func_name = ident.extract<Shader::Identifier>()->getValue();
                auto func_decl = CreateNodeTyped<FunctionDeclaration>(func_decl_loc, return_type,
                                                                      func_name,
                                                                      std::move(args));
                func_decl_ptr = func_decl.get();
                func = CreateNode<Reference<FunctionDeclaration>>(func_decl_loc, func_decl.get());
                auto func_set = m_Driver.createStackNode<FunctionSet>(func_decl_loc, func_name);
                func_set->pushFunction(std::move(func_decl));
            }
            else
            {
                auto func_set = ident.extract<FunctionSet>();
                func_decl_ptr = func_set->getFunction(args.extract<List>());
                if(!func_decl_ptr)
                {
                    auto func_decl_node = CreateNodeTyped<FunctionDeclaration>(func_decl_loc, return_type, func_set->getNodeName(), std::move(args));
                    func_decl_ptr = func_decl_node.get();
                    func_set->pushFunction(std::move(func_decl_node));
                }
                func = CreateNode<Reference<FunctionDeclaration>>(func_decl_loc, func_decl_ptr);
            }
        }
        m_NodeStack.emplace_back(std::move(func));
    }
    else if(IsCharacterToken(m_CurrentToken, '{'))
    {
        AST::Node statement_list;
        {
            m_Driver.beginBlock();
            auto& driver = m_Driver;
            auto at_exit = CreateAtScopeExit([&driver](){ driver.endBlock(); });
            parseToken();
            statement_list = statementList();
            auto status = expect(ToCharacterToken('}'));
            if(!status)
            {
                m_Driver.error(m_CurrentLocation, "Invalid function definition");
                m_NodeStack.emplace_back(AST::Node());
                return HANDLED;
            }
        }

        NodeT<FunctionDefinition> func;
        FunctionDeclaration* func_decl_ptr = nullptr;
        if(ident)
        {
            if(ident_type == ShaderToken::Identifier)
            {
                string func_name = ident.extract<Shader::Identifier>()->getValue();
                auto func_decl = CreateNodeTyped<FunctionDeclaration>(func_decl_loc, return_type,
                                                                      func_name,
                                                                      std::move(args));
                func = CreateNodeTyped<FunctionDefinition>(func_decl_loc, func_decl.get(), std::move(statement_list));
                func_decl_ptr = func_decl.get();
                auto func_set = m_Driver.createStackNode<FunctionSet>(func_decl_loc, func_name);
                func_set->pushFunction(std::move(func_decl));
            }
            else
            {
                auto func_set = ident.extract<FunctionSet>();
                func_decl_ptr = func_set->getFunction(args.extract<List>());
                if(!func_decl_ptr)
                {
                    auto func_decl_node = CreateNodeTyped<FunctionDeclaration>(func_decl_loc, return_type, func_set->getNodeName(), std::move(args));
                    func_decl_ptr = func_decl_node.get();
                    func_set->pushFunction(std::move(func_decl_node));
                }

                func = CreateNode<FunctionDefinition>(func_decl_loc, func_decl_ptr, std::move(args));
            }
        }

        m_NodeStack.emplace_back(std::move(func));
    }
    else
    {
        m_NodeStack.emplace_back(AST::Node());
        m_Driver.error(func_decl_loc, "Invalid function declaration");
    }

    return HANDLED;
}

bool Parser::expect(ShaderToken expected)
{
    parseToken();
    if(m_CurrentToken != expected)
    {
        m_CurrentNode = AST::Node();
        std::stringstream ss;
        ss << "Unexpected \"" << m_CurrentToken << "\". Expecting \"" << expected << "\" instead.";
        m_Driver.error(m_CurrentLocation, ss.str());
        return false;
    }
    return true;
}

AST::Node Parser::expectNode(ShaderToken expected)
{
    expect(expected);
    return std::move(m_CurrentNode);
}

AST::Node Parser::expectRedefCheck(ShaderToken expected, ShaderToken redef)
{
    parseToken();
    if(m_CurrentToken == expected)
        return std::move(m_CurrentNode);

    if(m_CurrentToken == redef)
    {
        m_CurrentNode = AST::Node();
        std::stringstream ss;
        ss << "Redefinition of " << m_CurrentNode.getNodeName() << ". Previously declared at: " << m_CurrentNode.getDeclarationLocation();
        m_Driver.error(m_CurrentLocation, ss.str());
        return AST::Node();
    }
    
    m_CurrentNode = AST::Node();
    std::stringstream ss;
    ss << "Unexpected \"" << m_CurrentToken << "\". Expecting \"" << expected << "\" instead.";
    m_Driver.error(m_CurrentLocation, ss.str());
    return AST::Node();
}

AST::NodeT<List> Parser::collapseStackToList(ListType _type, size_t top_element)
{
    if(top_element == m_NodeStack.size())
        return AST::Node();

    auto list_node = CreateNodeTyped<List>(m_CurrentLocation, _type, std::move(m_NodeStack.back()));
    for(auto cur_elem = m_NodeStack.size() - 1, last_elem = top_element; cur_elem-- > last_elem;)
    {
        list_node = CreateNodeTyped<List>(m_CurrentLocation, _type, std::move(m_NodeStack[cur_elem]), std::move(list_node));
    }
    m_NodeStack.resize(top_element);
    return std::move(list_node);
}

static bool IsOutputQualifier(ShaderToken token)
{
    return token == ShaderToken::InQualifier ||
           token == ShaderToken::OutQualifier ||
           token == ShaderToken::CentroidQualifier ||
           token == ShaderToken::SampleQualifier;
}

static bool IsInterpolationQualifier(ShaderToken token)
{
    return token == ShaderToken::FlatQualifier ||
           token == ShaderToken::NoPerspectiveQualifier ||
           token == ShaderToken::SmoothQualifier;
}

InterpolationQualifier TranslateInterpolationQualifier(ShaderToken token)
{
    switch(token)
    {
    default: TGE_ASSERT(false, "unknown interpolation");
    case ShaderToken::FlatQualifier: return InterpolationQualifier::Flat;
    case ShaderToken::NoPerspectiveQualifier: return InterpolationQualifier::Noperspective;
    case ShaderToken::SmoothQualifier: return InterpolationQualifier::Smooth;
    }
}

AST::Node Parser::layoutHeader()
{
    auto status = expect(ToCharacterToken('('));
    if(!status)
    {
        return AST::Node();
    }

    size_t top_element = m_NodeStack.size();

    for(;;)
    {
        auto layout_elem = m_CurrentLocation;
        auto ident = expectNode(ShaderToken::Identifier);
        if(!ident)
        {
            skipToToken(')');
            break;
        }

        parseToken();
        if(IsCharacterToken(m_CurrentToken, '='))
        {
            parseToken();
            if(m_CurrentToken != ShaderToken::Identifier &&
               m_CurrentToken != ShaderToken::Integer)
            {
                m_Driver.error(m_CurrentLocation, "Unexpected value type. Value types could be only integer and string identifiers");
                skipToToken(')');
                break;
            }
            m_NodeStack.emplace_back(CreateNode<BinaryOperator>(layout_elem, BinaryOperatorType::Assign, std::move(ident), std::move(m_CurrentNode)));
        }
        else
        {
            m_NodeStack.emplace_back(std::move(ident));
        }

        parseToken();
        if(IsCharacterToken(m_CurrentToken, ')'))
        {
            break;
        }
        else if(!IsCharacterToken(m_CurrentToken, ','))
        {
            m_Driver.error(m_CurrentLocation, "Unexpected token. List should be separated by ',' and contain data in the following form:\n"
                           "\"key\" or\n"
                           "\"key = value\"");
            skipToToken(')');
            break;
        }
    }

    return collapseStackToList(ListType::CommaSeparated, top_element);
}

bool Parser::globalVariable()
{
    StorageQualifier storage = StorageQualifier::Default;
    InterpolationQualifier interp = InterpolationQualifier::Default;
    bool invariant = false;
    
    AST::Node layout;

    if(m_CurrentToken != ShaderToken::LayoutQualifier &&
       m_CurrentToken != ShaderToken::InvariantQualifier &&
       m_CurrentToken != ShaderToken::ConstantQualifier &&
       !IsOutputQualifier(m_CurrentToken) &&
       !IsInterpolationQualifier(m_CurrentToken))
       return UNHANDLED;

    auto decl_loc = m_CurrentLocation;

    if(m_CurrentToken == ShaderToken::ConstantQualifier)
    {
        storage = StorageQualifier::Const;
    }
    else if(m_CurrentToken == ShaderToken::InvariantQualifier)
    {
        invariant = true;
    }
    else
    {
        if(m_CurrentToken == ShaderToken::LayoutQualifier)
        {
            layout = layoutHeader();
            if(!layout)
            {
                m_NodeStack.push_back(AST::Node());
                skipToToken(';');
                return AST::Node();
            }

            parseToken();
        }

        if(IsInterpolationQualifier(m_CurrentToken))
        {
            interp = TranslateInterpolationQualifier(m_CurrentToken);

            parseToken();
        }

        if(IsOutputQualifier(m_CurrentToken))
        {
            if(m_CurrentToken == ShaderToken::InQualifier)
            {
                storage = StorageQualifier::In;
            }
            else if(m_CurrentToken == ShaderToken::OutQualifier)
            {
                storage = StorageQualifier::Out;
            }
            else if(m_CurrentToken == ShaderToken::CentroidQualifier)
            {
                parseToken();
                if(m_CurrentToken == ShaderToken::InQualifier)
                {
                    storage = StorageQualifier::CentroidIn;
                }
                else if(m_CurrentToken == ShaderToken::OutQualifier)
                {
                    storage = StorageQualifier::CentroidOut;
                }
                else
                {
                    m_Driver.error(m_CurrentLocation, "Invalid output type");
                    skipToToken(';');
                    return HANDLED;
                }
            }
            else if(m_CurrentToken == ShaderToken::SampleQualifier)
            {
                parseToken();
                if(m_CurrentToken == ShaderToken::InQualifier)
                {
                    storage = StorageQualifier::SampleIn;
                }
                else if(m_CurrentToken == ShaderToken::OutQualifier)
                {
                    storage = StorageQualifier::SampleOut;
                }
                else
                {
                    m_Driver.error(m_CurrentLocation, "Invalid output type");
                    skipToToken(';');
                    return HANDLED;
                }
            }
            else
            {
                TGE_ASSERT(false, "broken");
            }

            parseToken();
        }
    }
    
    auto var_loc = m_CurrentLocation;
    auto var_node = variable();
    if(!var_node)
    {
        m_Driver.error(var_loc, "Invalid variable declaration");
        m_NodeStack.push_back(AST::Node());
        skipToToken(';');
        return HANDLED;
    }

    auto status = expect(ToCharacterToken(';'));
    if(!status)
    {
        m_Driver.error(decl_loc, "Invalid variable declaration");
        skipToToken(';');
        return HANDLED;
    }

    Variable* variable = var_node.extract<Variable>();
    variable->setInterpolation(interp);
    variable->setStorage(storage);
    variable->setLayout(std::move(layout));
    variable->setInvariant(invariant);

    m_NodeStack.emplace_back(CreateNode<Declaration>(decl_loc, std::move(var_node)));

    return HANDLED;
}

bool Parser::shader()
{
    ShaderType shader_type;

    auto shader_loc = m_CurrentLocation;

    switch(m_CurrentToken)
    {
    case ShaderToken::VertexQualifier: shader_type = ShaderType::VertexShader; break;
    case ShaderToken::GeometryQualifier: shader_type = ShaderType::GeometryShader; break;
    case ShaderToken::FragmentQualifier: shader_type = ShaderType::FragmentShader; break;
    default: return UNHANDLED;
    }

    auto status = expect(ShaderToken::Shader) &&
                  expect(ToCharacterToken('{'));
    if(!status)
    {
        skipDeclarationOrDefinition();
        m_NodeStack.push_back(AST::Node());
        return HANDLED;
    }

    parseToken();

    auto top_element = m_NodeStack.size();
    {
        m_Driver.beginShader(shader_type);
        auto& driver = m_Driver;
        auto at_exit = CreateAtScopeExit([shader_type, &driver]() { driver.endShader(); });

        for(;;)
        {
            auto loc = m_CurrentLocation;
            auto status = globalVariable()
                || function()
                // || invariantDeclaration() TODO
                || structDeclaration();
            // || optionalShaderExtDeclaration(); TODO

            if(!status)
            {
                m_Driver.error(loc, "Unexpected declaration");
                skipToToken('}');
                break;
            }

            parseToken();

            if(IsCharacterToken(m_CurrentToken, '}'))
                break;
        }
    }

    AST::Node child = collapseStackToList(ListType::SemicolonSeparated, top_element);
    m_NodeStack.emplace_back(CreateNode<Shader::ShaderDeclaration>(shader_loc, shader_type, std::move(child)));

    return HANDLED;
}

bool Parser::import()
{
    if(m_CurrentToken != ShaderToken::Import)
        return UNHANDLED;

    auto import_loc = m_CurrentLocation;

    auto string_literal = expectNode(ShaderToken::StringLiteral);
    
    if(!string_literal)
    {
        m_NodeStack.push_back(AST::Node());
        skipToToken(';');
        return HANDLED;
    }

    AST::Node child;

    parseToken();
    if(!IsCharacterToken(m_CurrentToken, ';'))
    {
        bool status = expect(ToCharacterToken('{'));
        if(!status)
        {
            auto top_element = m_NodeStack.size();

            parseToken();

            for(;;)
            {
                if(m_CurrentToken != ShaderToken::Identifier)
                {
                    m_Driver.error(m_CurrentLocation, "Import statement definition field should start with identifier.");
                    status = false;
                    break;
                }
                auto def_loc = m_CurrentLocation;
                auto ident = std::move(m_CurrentNode);

                status |= expect(ToCharacterToken('='));
                if(!status)
                    break;
                parseToken();
                if(m_CurrentToken != ShaderToken::Integer &&
                   m_CurrentToken != ShaderToken::Unsigned &&
                   m_CurrentToken != ShaderToken::Float &&
                   m_CurrentToken != ShaderToken::Boolean)
                {
                    m_Driver.error(m_CurrentLocation, "Invalid definition value. Valid values types are: integer, floating-point, boolean.");
                    status = false;
                    break;
                }

                parseToken();
                if(IsCharacterToken(m_CurrentToken, '}'))
                    break;

                m_NodeStack.emplace_back(CreateNode<BinaryOperator>(def_loc, BinaryOperatorType::Assign, std::move(ident), std::move(m_CurrentNode)));
            }

            child = collapseStackToList(ListType::CommaSeparated, top_element);
        }

        if(!status)
        {
            m_NodeStack.push_back(AST::Node());
            skipToToken(';');
            return HANDLED;
        }
    }

    m_NodeStack.emplace_back(CreateNode<Import>(import_loc, "\"" + string_literal.extract<Shader::StringLiteral>()->getValue() + "\"", std::move(child)));

    return HANDLED;
}

bool Parser::structDeclaration()
{
    if(m_CurrentToken != ShaderToken::StructQualifier)
        return UNHANDLED;

    auto struct_decl_loc = m_CurrentLocation;

    auto ident = expectNode(ShaderToken::Identifier);
    if(!ident)
    {
        m_NodeStack.push_back(AST::Node());
        skipToToken('}');
        skipToToken(';');
        return HANDLED;
    }

    bool status = expect(ToCharacterToken('{'));
    if(!status)
    {
        m_NodeStack.push_back(AST::Node());
        skipToToken('}');
        skipToToken(';');
        return HANDLED;
    }

    parseToken();

    auto top_element = m_NodeStack.size();
    {
        m_Driver.beginBlock();
        auto& driver = m_Driver;
        auto at_exit = CreateAtScopeExit([&driver]() { driver.endBlock(); });

        for(;;)
        {
            auto var_loc = m_CurrentLocation;
            auto status = basicVariableDeclaration();

            if(!status)
            {
                m_NodeStack.push_back(AST::Node());
                skipToToken('}');
                skipToToken(';');
                return HANDLED;
            }

            parseToken();

            if(IsCharacterToken(m_CurrentToken, '}'))
                break;
        }

        status = expect(ToCharacterToken(';'));
        if(!status)
        {
            m_NodeStack.push_back(AST::Node());
            skipToToken(';');
            return HANDLED;
        }
    }

    AST::Node child = collapseStackToList(ListType::SemicolonSeparated, top_element);
    m_NodeStack.emplace_back(CreateNode<Declaration>(struct_decl_loc, m_Driver.createStackType<StructType>(struct_decl_loc, ident ? ident.extract<Identifier>()->getValue() : string(), std::move(child))));

    return HANDLED;
}

void Parser::skipDefinition()
{
    size_t block = 1;
    auto loc = m_CurrentLocation;
    for(;;)
    {
        if(IsCharacterToken(m_CurrentToken, '{'))
        {
            ++block;
        }
        else if(IsCharacterToken(m_CurrentToken, '}'))
        {
            --block;
            if(block == 0)
                break;
        }
        else if(m_CurrentToken == ShaderToken::EndOfFile)
        {
            m_Driver.error(loc, string("Reached end of file before reaching }"));
            return;
        }
        parseToken();
    }
}

void Parser::skipDeclarationOrDefinition()
{
    size_t block = 0;
    auto loc = m_CurrentLocation;
    while(block || m_CurrentToken != ToCharacterToken(';'))
    {
        if(block && IsCharacterToken(m_CurrentToken, '}'))
        {
            --block;
            if(block == 0)
                break;
        }
        if(IsCharacterToken(m_CurrentToken, '{'))
        {
            ++block;
        }
        else if(m_CurrentToken == ShaderToken::EndOfFile)
        {
            m_Driver.error(loc, string("Reached end of file before reaching ") + (block ? '}' : ';'));
            return;
        }
        parseToken();
    }
}

void Parser::skipToToken(char _c)
{
    auto loc = m_CurrentLocation;
    while(m_CurrentToken != ToCharacterToken(_c))
    {
        if(m_CurrentToken == ShaderToken::EndOfFile)
        {
            m_Driver.error(loc, string("Reached end of file before reaching ") + _c);
            return;
        }
        parseToken();
    }
}

bool Parser::bufferDeclaration()
{
    BufferType buf_type;
    bool status = true;
    auto buf_decl_loc = m_CurrentLocation;
    switch(m_CurrentToken)
    {
    case ShaderToken::BufferQualifier: buf_type = BufferType::Regular; break;
    case ShaderToken::ConstantQualifier:
    {
        buf_type = BufferType::Constant;
        status = expect(ShaderToken::BufferQualifier);
    } break;
    case ShaderToken::ResourceQualifier:
    {
        buf_type = BufferType::Resource;
        status = expect(ShaderToken::BufferQualifier);
    } break;
    default:
        return UNHANDLED;
    }

    if(!status)
    {
        m_NodeStack.emplace_back(AST::Node());
        skipToToken('}');
        return HANDLED;
    }

    auto ident = expectNode(ShaderToken::Identifier);
    if(!ident)
    {
        m_NodeStack.emplace_back(AST::Node());
        skipToToken('}');
        return HANDLED;
    }

    status = expect(ToCharacterToken('{'));
    if(!status)
    {
        m_NodeStack.emplace_back(AST::Node());
        skipToToken('}');
        return HANDLED;
    }

    parseToken();

    auto top_element = m_NodeStack.size();

    for(;;)
    {
        AST::Node layout;
        if(m_CurrentToken == ShaderToken::LayoutQualifier)
        {
            layout = layoutHeader();
            if(layout)
            {
                parseToken();
            }
        }

        auto var_loc = m_CurrentLocation;
        AST::Node var_node = variable();
        if(var_node)
        {
            var_node.extract<Variable>()->setLayout(std::move(layout));
            m_NodeStack.emplace_back(CreateNode<Declaration>(var_loc, std::move(var_node)));
        }
        else
        {
            m_Driver.error(var_loc, "Invalid declaration. Buffers should be comprised of variables.");
            skipToToken(';');
            m_NodeStack.emplace_back(AST::Node());
        }

        status = expect(ToCharacterToken(';'));
        if(!status)
        {
            m_Driver.error(var_loc, "Variable declaration should end with ;");
            skipToToken(';');
        }

        parseToken();
        if(IsCharacterToken(m_CurrentToken, '}'))
            break;
    }
    
    AST::Node child = collapseStackToList(ListType::SemicolonSeparated, top_element);

    auto buffer = CreateNodeTyped<Shader::Buffer>(buf_decl_loc, ident.extract<Identifier>()->getValue(), std::move(child));
    m_NodeStack.emplace_back(std::move(buffer));

    return HANDLED;
}

bool Parser::structBufferDeclaration()
{
    if(m_CurrentToken != ShaderToken::StructBufferQualifier)
        return UNHANDLED;

    auto struct_buf_loc = m_CurrentLocation;

    auto type_node = expectNode(ShaderToken::Type);
    if(!type_node)
    {
        m_NodeStack.push_back(AST::Node());
        return HANDLED;
    }

    // TODO: Add redefinition
    auto ident = expectNode(ShaderToken::Identifier);
    if(!ident)
    {
        m_NodeStack.push_back(AST::Node());
        return HANDLED;
    }

    bool status = expect(ToCharacterToken(';'));
    if(!status)
    {
        m_NodeStack.push_back(AST::Node());
        skipToToken(';');
        return HANDLED;
    }

    auto* _type = type_node.extract<Shader::Type>();

    AST::NodeT<VariableRef> var;
    if(_type->getTypeEnum() == Shader::ElementType::Struct)
    {
        auto arr_type = m_Driver.createInternalType<ArrayType>(struct_buf_loc, _type, AST::Node());
        var = m_Driver.createStackNode<Variable>(struct_buf_loc, arr_type.get(), ident.extract<Shader::Identifier>()->getValue());
        var->setStorage(StorageQualifier::StructBuffer);
    }
    else
    {
        m_Driver.error(struct_buf_loc, _type->getNodeName() + " is not struct type. structbuffer is limited to struct types only.");
    }

    m_NodeStack.emplace_back(CreateNode<Declaration>(struct_buf_loc, std::move(var)));

    return HANDLED;
}

bool Parser::buffer()
{
    return bufferDeclaration()
        || structBufferDeclaration();
}

bool Parser::optionalExtDeclaration()
{
    if(!IsCharacterToken(m_CurrentToken, '@'))
        return UNHANDLED;

    auto start_loc = m_CurrentLocation;
    auto opt = expectNode(ShaderToken::Option);
    if(!opt)
    {
        m_NodeStack.push_back(AST::Node());
        return HANDLED;
    }

    AST::Node child;
    {
        m_Driver.beginOptionBlock(m_CurrentNode.extract<Shader::Option>());
        auto& driver = m_Driver;
        auto at_exit = CreateAtScopeExit([&driver]() { driver.endOptionBlock(); });

        parseToken();
        if(IsCharacterToken(m_CurrentToken, '{'))
        {
            auto top_element = m_NodeStack.size();
            bool status = true;
            Location decl_loc;
            for(;;)
            {
                parseToken();
                decl_loc = m_CurrentLocation;
                status &= removableExtDeclaration();
                if(!status)
                    break;
            }
            if(!status)
            {
                m_Driver.error(decl_loc, "Invalid expression");
                skipDefinition();
            }

            child = collapseStackToList(ListType::SemicolonSeparated, top_element);
        }
        else
        {
            bool status = removableExtDeclaration();
            if(status)
            {
                child = std::move(m_NodeStack.back());
                m_NodeStack.pop_back();
            }
        }
    }

    auto optional = CreateNode<Optional>(start_loc, opt.extract<Option>(), std::move(child));
    m_NodeStack.emplace_back(std::move(optional));

    return HANDLED;
}

bool Parser::removableExtDeclaration()
{
    return import()
        || buffer()
        || structDeclaration()
        || shader()
        || function()
        || optionalExtDeclaration();
}

bool Parser::options()
{
    if(m_CurrentToken != ShaderToken::Options)
        return UNHANDLED;
    parseToken();
    if(!expect(ToCharacterToken('{')))
    {
        std::stringstream ss;
        ss << "Invald options block. Options blocks should start with \"options\" keyword followed by {.\n"
              "It was followed by " << m_CurrentToken << " instead.";
        m_Driver.error(m_CurrentLocation, ss.str());
        m_NodeStack.push_back(AST::Node());
        skipToToken('}');
        return HANDLED;
    }

    auto opt_node = CreateNodeTyped<OptionsDeclaration>(m_CurrentLocation);
    auto* _options = opt_node.get();
    m_NodeStack.push_back(std::move(opt_node));

    auto stack_size = m_NodeStack.size();

    for(;;)
    {
        auto ident = expectNode(ShaderToken::Identifier);
        if(!ident)
        {
            skipToToken('}');
            break;
        }

        {
            auto option = m_Driver.createStackNode<Option>(m_CurrentLocation, ident.extract<Value<string>>()->getValue());
            TGE_ASSERT(stack_size == m_NodeStack.size(), "Stack should not expand");
            _options->addOption(option.get());
        }

        parseToken();
        if(IsCharacterToken(m_CurrentToken, '}'))
            break;
        bool status = expect(ToCharacterToken(','));
        if(!status)
        {
            skipToToken('}');
            break;
        }
    }

    return HANDLED;
}

int Parser::parse()
{
    parseToken();

    while(m_CurrentToken != ShaderToken::EndOfFile)
    {
        auto loc = m_CurrentLocation;
        auto status =  removableExtDeclaration()
                    || options();
        if(!status)
        {
            m_Driver.error(loc, "Unexpected statement");
            break;
        }

        parseToken();
    }

    m_Driver.setASTRoot(collapseStackToList(ListType::SemicolonSeparated, 0));

    return 0;
}
}
}