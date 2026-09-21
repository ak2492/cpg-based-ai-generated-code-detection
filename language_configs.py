"""Per-language constants, exactly as defined in the three notebooks.

Shared items (RE_SNAKE/CAMEL/PASCAL/TACTICAL, LAZY_IDENTIFIERS,
MAGIC_NUM_EXCLUSIONS, MAX_*, continuous_idx, SEED) are byte-identical
across notebooks and therefore defined once here. Language-specific
sets are kept separate so outputs never change.
"""
import re

SEED = 42
MAX_VOCAB_SIZE, BPE_VOCAB_SIZE, MAX_SUBWORDS = 350, 6000, 12
MAX_AST_DEPTH, MAX_AST_NODES, NUM_RELATIONS = 150, 2200, 16
MAX_SEMEVAL_SAMPLES_PER_CLASS = 3000

CONTINUOUS_IDX = [0, 1, 5, 6, 7, 8, 9, 15, 16, 22, 24, 29, 30]

# Shared (identical in all three notebooks)
RE_SNAKE = re.compile(r'^[a-z0-9]+(_[a-z0-9]+)+$')
RE_CAMEL = re.compile(r'^[a-z]+([A-Z][a-z0-9]+)+$')
RE_PASCAL = re.compile(r'^([A-Z][a-z0-9]+)+$')
RE_TACTICAL = re.compile(r'\b(todo|fixme|hack|note|bug)\b', re.IGNORECASE)
LAZY_IDENTIFIERS = {'tmp', 'temp', 'ans', 'res', 'val', 'flag', 'cnt', 'cur', 'foo', 'bar', 'data', 'obj'}
MAGIC_NUM_EXCLUSIONS = {'0', '1', '2', '-1', '10', '0.0', '1.0'}

PYTHON_RESERVED = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    "self", "cls", "int", "str", "float", "bool", "list", "dict", "set",
    "tuple", "print", "range", "len", "enumerate", "zip", "open"
}

JAVA_RESERVED = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class",
    "const", "continue", "default", "do", "double", "else", "enum", "extends", "final",
    "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
    "throw", "throws", "transient", "try", "void", "volatile", "while", "true", "false",
    "null", "String", "System", "out", "println", "print", "main", "Override"
}

CPP_RESERVED = {
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
    "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
    "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
    "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
    "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
    "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace",
    "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
    "private", "protected", "public", "reflexpr", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static", "static_assert",
    "static_cast", "struct", "switch", "synchronized", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
    "while", "xor", "xor_eq", "std", "cout", "cin", "endl", "vector", "string",
    "include", "define", "main", "pair", "map", "set", "unordered_map"
}

TEXT_CAPTURE_TYPES = {
    "python": {
        'identifier', 'type_identifier', 'primitive_type', 'string_literal', 'string',
        'char_literal', 'number_literal', 'integer', 'float', 'comment', 'preproc_def', 'preproc_include'
    },
    "java": {
        'identifier', 'type_identifier', 'string_literal', 'character_literal',
        'decimal_integer_literal', 'hex_integer_literal', 'decimal_floating_point_literal',
        'comment', 'block_comment', 'line_comment'
    },
    "cpp": {
        'identifier', 'type_identifier', 'primitive_type', 'string_literal',
        'char_literal', 'number_literal', 'comment', 'preproc_def', 'preproc_include'
    },
}

SCOPE_TYPES = {
    "python": {
        'function_definition', 'class_specifier', 'class_definition',
        'struct_specifier', 'namespace_definition', 'lambda_expression', 'template_declaration'
    },
    "java": {
        'class_declaration', 'method_declaration', 'interface_declaration',
        'enum_declaration', 'constructor_declaration', 'record_declaration'
    },
    "cpp": {
        'function_definition', 'class_specifier', 'struct_specifier',
        'namespace_definition', 'lambda_expression', 'template_declaration'
    },
}

CONTROL_TRIGGERS = {
    "python": {
        'if_statement', 'for_statement', 'while_statement', 'do_statement',
        'switch_statement', 'case_statement', 'try_statement', 'catch_clause', 'except_clause',
        'return_statement', 'break_statement', 'continue_statement',
        'goto_statement', 'throw_statement', 'raise_statement', 'new_expression', 'delete_expression', 'yield'
    },
    "java": {
        'if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement', 'do_statement',
        'switch_expression', 'switch_statement', 'try_statement', 'catch_clause', 'finally_clause',
        'return_statement', 'break_statement', 'continue_statement', 'throw_statement', 'yield_statement'
    },
    "cpp": {
        'if_statement', 'for_statement', 'while_statement', 'do_statement',
        'switch_statement', 'case_statement', 'try_statement', 'catch_clause',
        'return_statement', 'break_statement', 'continue_statement',
        'goto_statement', 'throw_statement', 'new_expression', 'delete_expression'
    },
}

DECISION_TYPES = {
    "python": {'if_statement', 'for_statement', 'while_statement', 'do_statement', 'case_statement', 'conditional_expression'},
    "java": {'if_statement', 'for_statement', 'enhanced_for_statement', 'while_statement', 'do_statement', 'switch_statement', 'ternary_expression'},
    "cpp": {'if_statement', 'for_statement', 'while_statement', 'do_statement', 'case_statement', 'conditional_expression'},
}

DEFAULT_BATCH_SIZE = {"python": 128, "java": 32, "cpp": 16}
ACCUMULATION_STEPS = 2

CLEAN_CHECKPOINT = {
    "python": "model_python_clean_baseline_checkpoint.pth",
    "cpp": "model_cpp_clean_baseline_checkpoint.pth",
    "java": "model_clean_baseline_checkpoint.pth",
}
ADV_CHECKPOINT = {
    "python": "model_python_adv_augmented_checkpoint.pth",
    "cpp": "model_cpp_adv_augmented_checkpoint.pth",
    "java": "model_adv_augmented_checkpoint.pth",
}

MACRO_NAMES = [
    "Case Consistency", "Indent Step Var", "Indent Variance", "Comment Ratio",
    "Tactical Comment Ratio", "Max Nesting Depth", "Avg Nesting Depth", "Line Len Variance",
    "Long Line Ratio", "Import Density", "Blank Line Entropy", "Gzip Ratio",
    "Halstead Volume Log", "Func Max/Mean Ratio", "Avg Param Count", "Call/Def Ratio"
]


def get_parser(language):
    """Instantiate tree-sitter parser exactly as notebooks do: Language(tsX.language()) + Parser(LANG)."""
    from tree_sitter import Language, Parser
    if language == "python":
        import tree_sitter_python as tsmod
        lang = Language(tsmod.language())
    elif language == "java":
        import tree_sitter_java as tsmod
        lang = Language(tsmod.language())
    elif language == "cpp":
        import tree_sitter_cpp as tsmod
        lang = Language(tsmod.language())
    else:
        raise ValueError(f"Unsupported language: {language}")
    return Parser(lang), lang


def get_reserved(language):
    return {"python": PYTHON_RESERVED, "java": JAVA_RESERVED, "cpp": CPP_RESERVED}[language]
