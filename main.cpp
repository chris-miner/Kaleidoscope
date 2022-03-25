//
//  main.cpp
//  toy language
//
//  Created by Chris Miner on 3/20/22.
//

// TODO: why does this header have code in it?
#include "KaleidoscopeJIT.h"

// MARK: - IR Generation Includes
#include "llvm/IR/Value.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

// MARK: - Optimization Manager & Passes
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/TargetSelect.h"

// MARK: - c++ standard templating library
#include <string>
#include <vector>
#include <map>

using namespace llvm;
using namespace llvm::orc;

// MARK: - Lexer Code Goes Here
//  The lexer returns tokens [0-255] if it is an unknown character, otherwise one
//  of these for known things.
enum Token
{
    tok_eof = -1, // end of message token

    // commands
    tok_def = -2,    // def keyword token
    tok_extern = -3, // extern keyword token

    // primary
    tok_identifier = -4, // alphanumeric label token
    tok_number = -5,     // decimal number literal token
};

static std::string IdentifierStr; // Filled in if tok_identifier
static double NumVal;             // Filled in if tok_number

// gettok - Return the next token from standard input.
// tokens are number, identifier, keyword 'def', keyword 'extern', EOF, and single characters like '(', '+', ')', etc.
static int gettok()
{
    static int LastChar = ' ';

    // Skip any whitespace.
    while (isspace(LastChar))
        LastChar = getchar();

    // looking for an identifier: [a-zA-Z][a-zA-Z0-9]*
    if (isalpha(LastChar))
    {
        IdentifierStr = LastChar;
        // identifiers are a string of numbers or letters
        while (isalnum((LastChar = getchar())))
            IdentifierStr += LastChar; // accumulate the identifier in global

        // special identifier keywords
        if (IdentifierStr == "def")
            return tok_def;
        if (IdentifierStr == "extern")
            return tok_extern;
        return tok_identifier;
    }

    // looking for a Number: [0-9.]+
    if (isdigit(LastChar) || LastChar == '.')
    {
        std::string NumStr;
        do
        {
            NumStr += LastChar; // accumulate the number in local
            LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        NumVal = strtod(NumStr.c_str(), 0); // 'return' number as global
        return tok_number;
    }

    // looking for a Comment: [#]+.*
    if (LastChar == '#')
    {
        do
            LastChar = getchar();
        // skip rest of line
        while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        // if last char is EOF, we want to return that as a token
        if (LastChar != EOF)
            return gettok();
    }

    // Check for end of file.  Don't consume the EOF.
    if (LastChar == EOF)
        return tok_eof;

    // Otherwise, just return the character as its ascii value.
    int ThisChar = LastChar;
    LastChar = getchar();
    return ThisChar;
}

// MARK: - Parser Code Goes Here

/*
 Classes for representing the Abstract Syntax Tree (AST) nodes
 These classes are wrapped in an anonymous namespace so they are
 only visible within this file.  This is like declaring a static
 variable or a static function.
 */
namespace
{

    // ExprAST - Base class for all expression nodes.
    class ExprAST
    {
    public:
        virtual ~ExprAST() {}
        /// codegen() emits IR for this AST node along with dependencies
        /// it returns an LLVM Value object
        /// The LLVM Value class represents a Static Single Assignment (SSA) register
        /// Normally code generation would be done with a visitor walking the AST
        /// todo: lookup SSA
        virtual Value *codegen() = 0;
    };

    // NumberExprAST - Expression class for numeric literals like "1.0".
    class NumberExprAST : public ExprAST
    {
        double Val; // stores value 'returned' by lexer

    public:
        NumberExprAST(double Val) : Val(Val) {}
        Value *codegen();
    };

    // VariableExprAST - Expression class for referencing a variable, like "a".
    // why doesn't the variale expression have value?
    class VariableExprAST : public ExprAST
    {
        std::string Name;

    public:
        VariableExprAST(const std::string &Name) : Name(Name) {}
        Value *codegen();
    };

    // BinaryExprAST - Expression class for a binary operator.
    // this is for things like [+-*/]
    class BinaryExprAST : public ExprAST
    {
        char Op;
        std::unique_ptr<ExprAST> LHS, RHS;

    public:
        BinaryExprAST(char op, std::unique_ptr<ExprAST> LHS,
                      std::unique_ptr<ExprAST> RHS)
            : Op(op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
        Value *codegen();
    };

    // CallExprAST - Expression class for function calls.
    class CallExprAST : public ExprAST
    {
        std::string Callee;
        std::vector<std::unique_ptr<ExprAST>> Args;

    public:
        CallExprAST(const std::string &Callee,
                    std::vector<std::unique_ptr<ExprAST>> Args)
            : Callee(Callee), Args(std::move(Args)) {}
        Value *codegen();
    };

    // PrototypeAST - This class represents the "prototype" for a function,
    // which captures its name, and its argument names (thus implicitly the number
    // of arguments the function takes).
    class PrototypeAST
    {
        std::string Name;
        std::vector<std::string> Args;

    public:
        PrototypeAST(const std::string &name, std::vector<std::string> Args)
            : Name(name), Args(std::move(Args)) {}

        const std::string &getName() const { return Name; }

        Function *codegen();
    };

    // FunctionAST - This class represents a function definition itself.
    class FunctionAST
    {
        std::unique_ptr<PrototypeAST> Proto;
        std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                    std::unique_ptr<ExprAST> Body)
            : Proto(std::move(Proto)), Body(std::move(Body)) {}
        Function *codegen();
    };
} // end anonymous namespace

// MARK: - Parser Helper functions

// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
// token the parser is looking at.  getNextToken reads another token from the
// lexer and updates global CurTok with its results.
static int CurTok;
static int getNextToken()
{
    return CurTok = gettok();
}

// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char *Str)
{
    fprintf(stderr, "LogError: %s\n", Str);
    return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char *Str)
{
    LogError(Str);
    return nullptr;
}

/*
 Basic Expression Parsing
 */
static std::unique_ptr<ExprAST> ParseExpression();

// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr()
{
    // we happen to know that CurTok points to a simple number token
    // so we create an AST node for it
    auto Result = std::make_unique<NumberExprAST>(NumVal);

    // consume the number token from lexer
    getNextToken();

    // return the new AST number node
    return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr()
{
    // we happen to know CurTok points at a '(' token
    // so we consume '(' token
    getNextToken();

    // parse the nested expression
    auto V = ParseExpression();
    if (!V)
        return nullptr;

    // expect ')' token to close parenexpr report error as needed
    if (CurTok != ')')
        return LogError("expected ')'");

    // comsume the ')' token from lexer
    getNextToken(); // consume ).

    // return the nested expression
    return V;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr()
{
    // assume CurTok is an identifier token and stash it.
    // the identifier refers to either a simple variable or function call
    std::string IdName = IdentifierStr;

    // consume identifier token and check for '(' token to follow.
    // if there is no '(' token then we have a plain-old variable identifier.
    getNextToken();
    if (CurTok != '(')
        // return the variable reference node
        return std::make_unique<VariableExprAST>(IdName);

    // this is an argument list
    // consume leading '(' token and expect 0 or more expressions
    getNextToken();

    // collect the 0 or more expressions separated by ','
    std::vector<std::unique_ptr<ExprAST>> Args;
    if (CurTok != ')')
    {
        while (1)
        {
            if (auto Arg = ParseExpression())
                Args.push_back(std::move(Arg)); // accumulate arguments
            else
                return nullptr; // error

            if (CurTok == ')')
                break;

            if (CurTok != ',')
                return LogError("Expected ')' or ',' in argument list");

            getNextToken();
        }
    }

    // consume the final ')' token.
    // fixme: this should be in the while loop.
    getNextToken();

    // we parsed an identifierexpression that is actually a function call
    return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static std::unique_ptr<ExprAST> ParsePrimary()
{
    switch (CurTok)
    {
    default:
        return LogError("unknown token when expecting an expression");
    case tok_identifier:
        return ParseIdentifierExpr();
    case tok_number:
        return ParseNumberExpr();
    case '(':
        return ParseParenExpr();
    }
}

// MARK: - Binary Expression Parsing

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.  (eg <, +, -, *)
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence()
{
    // I'm not sure what the point of this is.
    // maybe it is because our Map has a char key and we can't use non ascii chars to index it?
    if (!isascii(CurTok))
        return -1;

    // Make sure it's a declared binop.
    int TokPrec = BinopPrecedence[CurTok];

    // I can't see how this would ever be less than one.
    if (TokPrec <= 0)
        return -1;

    // so ... we are just looking up in the map
    // and returning -1 if the key is non ascii or ???
    return TokPrec;
}

/// binoprhs
///   ::= ('+' primary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS)
{
    // If this is a binop, find its precedence.
    while (true)
    {
        int TokPrec = GetTokPrecedence();

        // If this is a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (TokPrec < ExprPrec)
            return LHS;

        // Okay, we know this is a binop.
        int BinOp = CurTok;
        getNextToken(); // consume binop

        // Parse the primary expression after the binary operator.
        auto RHS = ParsePrimary();
        if (!RHS)
            return nullptr;

        // If BinOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as its LHS.
        int NextPrec = GetTokPrecedence();
        if (TokPrec < NextPrec)
        {
            RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
            if (!RHS)
                return nullptr;
        }

        // Merge LHS/RHS.
        LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
    }
}

/// expression
///   ::= primary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression()
{
    auto LHS = ParsePrimary();
    if (!LHS)
        return nullptr;

    return ParseBinOpRHS(0, std::move(LHS));
}

// MARK: - Parsing the Rest

/// prototype
///   ::= id '(' id* ')'
static std::unique_ptr<PrototypeAST> ParsePrototype()
{
    if (CurTok != tok_identifier)
        return LogErrorP("Expected function name in prototype");

    std::string FnName = IdentifierStr;
    getNextToken();

    if (CurTok != '(')
        return LogErrorP("Expected '(' in prototype");

    std::vector<std::string> ArgNames;
    while (getNextToken() == tok_identifier)
        ArgNames.push_back(IdentifierStr);
    if (CurTok != ')')
        return LogErrorP("Expected ')' in prototype");

    // success.
    getNextToken(); // consume ')'.

    return std::make_unique<PrototypeAST>(FnName, std::move(ArgNames));
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition()
{
    getNextToken(); // consume def.
    auto Proto = ParsePrototype();
    if (!Proto)
        return nullptr;

    if (auto E = ParseExpression())
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    return nullptr;
}

/// external ::= 'extern' prototype
/// supports declaration of functions both those that are built-in like sin and cos,
/// but also forward declarations of our own functions
static std::unique_ptr<PrototypeAST> ParseExtern()
{
    getNextToken(); // consume 'extern' keyword token.
    return ParsePrototype();
}

/// toplevelexpr ::= expression
/// this amounts to the whole program
/// wrap the whole program in an anonymous function
static std::unique_ptr<FunctionAST> ParseTopLevelExpr()
{
    if (auto E = ParseExpression())
    {
        // Make an anonymous proto.
        auto Proto = std::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

// MARK: - Code Generation Setup

// Context is opaque and owns a lot of core LLVM data structures
// such as type and constant value tables
// static LLVMContext TheContext;
static std::unique_ptr<LLVMContext> TheContext;

// Module is an LLVM object that tracks functions and global variables
// owns memory of generated IR
static std::unique_ptr<Module> TheModule;

// helper object making it easy to generate LLVM instructions
// tracks current place to insert instructions and creates instructions
// static IRBuilder<> Builder(TheContext);
static std::unique_ptr<IRBuilder<>> Builder;

// tracks which values are defined in the current scope and their LLVM rep
// this is the symbol table for the code.  for example function parameters
// when generating function bodies.
static std::map<std::string, Value *> NamedValues;

// holds and organizes the LLVM optimizations that we want to run.
// Each module requires its own function pass manager
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;

// TODO: What does this do?  It involves some methods declared in a header file.
static std::unique_ptr<KaleidoscopeJIT> TheJIT;

// TODO: What does this line do?
static ExitOnError ExitOnErr;

// holds the most recent prototype for each function
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;

Value *LogErrorV(const char *Str)
{
    LogError(Str);
    return nullptr;
}

// MARK: - Expression Code Generation

Value *NumberExprAST::codegen()
{
    return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen()
{
    // Look this variable up in the function.
    Value *V = NamedValues[Name];
    if (!V)
        LogErrorV("Unknown variable name");
    return V;
}

Value *BinaryExprAST::codegen()
{
    Value *L = LHS->codegen();
    Value *R = RHS->codegen();
    if (!L || !R)
        return nullptr;

    switch (Op)
    {
    case '+':
        return Builder->CreateFAdd(L, R, "addtmp");
    case '-':
        return Builder->CreateFSub(L, R, "subtmp");
    case '*':
        return Builder->CreateFMul(L, R, "multmp");
    case '<':
        L = Builder->CreateFCmpULT(L, R, "cmptmp");
        // Convert bool 0/1 to double 0.0 or 1.0
        return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
    default:
        return LogErrorV("invalid binary operator");
    }
}

/// Searches TheModule for an existing function declaration, falling back to generating a new declaration from FunctionProtos if it doesn’t find one.
Function *getFunction(std::string Name)
{
    // First, see if the function has already been added to the current module.
    if (auto *F = TheModule->getFunction(Name))
        return F;

    // If not, check whether we can codegen the declaration from some existing
    // prototype.
    auto FI = FunctionProtos.find(Name);
    if (FI != FunctionProtos.end())
        return FI->second->codegen();

    // If no existing prototype exists, return null.
    return nullptr;
}

Value *CallExprAST::codegen()
{
    // Look up the name in the global module table.
    //    Function *CalleeF = TheModule->getFunction(Callee);
    Function *CalleeF = getFunction(Callee);

    if (!CalleeF)
        return LogErrorV("Unknown function referenced");

    // If argument mismatch error.
    if (CalleeF->arg_size() != Args.size())
        return LogErrorV("Incorrect # arguments passed");

    std::vector<Value *> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
    {
        ArgsV.push_back(Args[i]->codegen());
        if (!ArgsV.back())
            return nullptr;
    }

    return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

// MARK: - Function Code Generation

Function *PrototypeAST::codegen()
{
    // Make the function type:  double(double,double) etc.
    // all function arguments are double, so we need an
    // array with as many doubles as we have arguments
    std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*TheContext));

    // declare a function signature returning a double taking a number of arguments that are doubles
    FunctionType *FT = FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

    // finally create a function with the constructed type
    // external linkage means the function may be defined outside the current module
    // or is callable by functions outside the current module
    // basically it says the function is both imported and exported
    Function *F = Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

    // Set names for all arguments.
    unsigned Idx = 0;
    for (auto &Arg : F->args())
        Arg.setName(Args[Idx++]);

    // return our function with no body.  This is a function prototype in LLVM
    return F;
}

Function *FunctionAST::codegen()
{
    // First, check for an existing function from a previous 'extern' declaration.
    /*
     Function *TheFunction = TheModule->getFunction(Proto->getName());

     if (!TheFunction)
     TheFunction = Proto->codegen();
     */
    auto &P = *Proto;
    FunctionProtos[Proto->getName()] = std::move(Proto);
    Function *TheFunction = getFunction(P.getName());

    if (!TheFunction)
        return nullptr;

    if (!TheFunction->empty())
        return (Function *)LogErrorV("Function cannot be redefined.");

    // Create a new basic block to start insertion into.
    // Basic Blocks in LLVM define the control flow graph
    // we have no control flow, so our functions have only one block
    BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    // static std::map<std::string, Value *> NamedValues;
    NamedValues.clear();
    for (auto &Arg : TheFunction->args())
    {
        NamedValues[std::string(Arg.getName())] = &Arg;
    }

    if (Value *RetVal = Body->codegen())
    {
        // Finish off the function.
        Builder->CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        verifyFunction(*TheFunction);

        // Optimize the function.
        TheFPM->run(*TheFunction);

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();
    return nullptr;
}
// MARK: - Optimization Pass and Module Initialization

static void InitializeModuleAndPassManager()
{
    TheContext = std::make_unique<LLVMContext>();

    // Open a new module.
    TheModule = std::make_unique<Module>("my cool jit", *TheContext);
    TheModule->setDataLayout(TheJIT->getDataLayout());

    // Create a new builder for the module.
    Builder = std::make_unique<IRBuilder<>>(*TheContext);

    // Create a new pass manager attached to it.  Each module has its own pass manager
    TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

    /*
     These four LLVM passes are useful for cleaning up a wide variety of code.
     */
    // Do simple "peephole" optimizations and bit-twiddling optzns.
    // Combine instructions to form fewer, simple instructions.
    // This pass performs algebraic simplifications.
    TheFPM->add(createInstructionCombiningPass());

    // Reassociate expressions.
    // This pass reassociates commutative expressions in an order that is
    // designed to promote better constant propagation
    TheFPM->add(createReassociatePass());

    // Eliminate Common SubExpressions.
    // Eliminates fully and partially redundant instructions and loads
    TheFPM->add(createGVNPass());

    // Simplify the Control Flow Graph (CFG)
    // Merges and eliminates unreachable blocks, converts switches to lookup tables, etc.
    TheFPM->add(createCFGSimplificationPass());

    TheFPM->doInitialization();
}

// MARK: - Parsing Driver

/*
 The Driver
 The lexer code above finds the tokens
 The parser code above recognizes valid strings of tokens while building AST
 This
 */

/// 'def' keyword token indicates we have a function definition coming up
/// so we'll parse that
static void HandleDefinition()
{
    if (auto FnAST = ParseDefinition())
    {
        if (auto *FnIR = FnAST->codegen())
        {
            fprintf(stderr, "Parsed a function definition:");
            FnIR->print(errs());
            fprintf(stderr, "\n");

            // transfer the newly defined function to the JIT and open a new module
            ExitOnErr(TheJIT->addModule(ThreadSafeModule(std::move(TheModule), std::move(TheContext))));
            InitializeModuleAndPassManager();
        }
    }
    else
    {
        // Skip token for error recovery.
        getNextToken();
    }
}

/// 'extern' keyword token indicates we have an externally defined function declaration
/// so we'll parse that
static void HandleExtern()
{
    if (auto ProtoAST = ParseExtern())
    {
        if (auto *FnIR = ProtoAST->codegen())
        {
            fprintf(stderr, "Parsed an extern:");
            FnIR->print(errs());
            fprintf(stderr, "\n");

            // Add the prototype to FunctionProtos
            FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
        }
    }
    else
    {
        // Skip token for error recovery.
        getNextToken();
    }
}

/// everything else is an expression
static void HandleTopLevelExpression()
{
    // Evaluate a top-level expression into an anonymous function.
    if (auto FnAST = ParseTopLevelExpr())
    {
        if (auto FnIR = FnAST->codegen())
        {
            fprintf(stderr, "Parsed a top-level expr:\n");
            FnIR->print(errs());
            fprintf(stderr, "\n");

            // Create a ResourceTracker to track JIT'd memory allocated to our
            // anonymous expression -- that way we can free it after executing.
            auto RT = TheJIT->getMainJITDylib().createResourceTracker();

            auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
            ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
            InitializeModuleAndPassManager();

            // Search the JIT for the __anon_expr symbol.
            auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));

            // Get the symbol's address and cast it to the right type (takes no
            // arguments, returns a double) so we can call it as a native function.
            double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();
            fprintf(stderr, "Evaluated to %f\n", FP());

            // Delete the anonymous expression module from the JIT.
            ExitOnErr(RT->remove());
        }
    }
    else
    {
        // Skip token for error recovery.
        getNextToken();
    }
}

/// a valid string is a definition, an external, an expression or a semi-colon
/// top ::= definition | external | expression | ';'|EOF
/// Our top-level expressions start with EOF, ';' , 'def', 'extern' or ???
static void MainLoop()
{
    while (1)
    {
        fprintf(stderr, "ready> ");
        switch (CurTok)
        {
        case tok_eof:
            return;
        case ';': // ignore top-level semicolons.
            getNextToken();
            break;
        case tok_def:
            HandleDefinition();
            break;
        case tok_extern:
            HandleExtern();
            break;
        default:
            HandleTopLevelExpression();
            break;
        }
    }
}

/// putchard - putchar that takes a double and returns 0.
extern "C" double putchard(double X)
{
    fputc((char)X, stderr);
    return 0;
}

// MARK: - Main Function
int main()
{
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();

    // setup binary operator order of operations
    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 20;
    BinopPrecedence['*'] = 40; // highest.

    // Prime the first token.
    fprintf(stderr, "ready> ");
    getNextToken(); // stashes token in CurTok global

    // create a Just In Time simple compiler
    TheJIT = ExitOnErr(KaleidoscopeJIT::Create());

    // Make the module, which holds all the code.
    // Open a new context and module.
    InitializeModuleAndPassManager();

    // run the REPL
    MainLoop();

    // Print out all of the generated code.
    //    TheModule->print(errs(), nullptr);
    return 0;
}
