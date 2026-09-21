"""CPG / AST graph construction — combined across python/java/cpp.

Common pieces (shannon_entropy, vocab fitting loop skeleton, process_split,
normalization, virtual node, 16 edge relations) are shared. Language-varying
pieces preserve each notebook byte-for-byte via `language` dispatch, so every
output is identical to its source notebook:

Edge relations (NUM_RELATIONS=16, all languages):
  0/1 parent<->child, 2/3 siblings, 4/5 same-text sequential uses,
  6/7 leaf-order chain, 8 node->virtual, 9/10 def<->use,
  11 CFG-next, 12/13 call<->def, 14/15 arg<->param.
"""
import sys
import math
import gzip
import re
from collections import Counter

import numpy as np
import torch
from tqdm.auto import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch_geometric.data import Data

from language_configs import (
    MAX_VOCAB_SIZE, BPE_VOCAB_SIZE, MAX_SUBWORDS,
    MAX_AST_DEPTH, MAX_AST_NODES, CONTINUOUS_IDX,
    TEXT_CAPTURE_TYPES, SCOPE_TYPES, CONTROL_TRIGGERS, DECISION_TYPES,
    RE_SNAKE, RE_CAMEL, RE_PASCAL, RE_TACTICAL,
    LAZY_IDENTIFIERS, MAGIC_NUM_EXCLUSIONS,
)

sys.setrecursionlimit(10000)


def shannon_entropy(s):
    return float(-sum((c / len(s)) * math.log2(c / len(s)) for c in Counter(s).values())) if s else 0.0


# ---------------------------------------------------------------------------
# Node stylometry (34-d). Bodies are exact copies of each notebook; dispatcher
# combines them under one signature without changing any output.
# ---------------------------------------------------------------------------

def _stylometry_python(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type):
    from language_configs import CONTROL_TRIGGERS as _CT
    control_triggers = _CT["python"]
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    line_span = float(end_row - start_row + 1)
    char_len = float(node.end_byte - node.start_byte)
    line_text = source_lines[start_row] if start_row < len(source_lines) else ""

    is_snake = 1.0 if text and RE_SNAKE.match(text) else 0.0
    is_camel = 1.0 if text and RE_CAMEL.match(text) else 0.0
    is_pascal = 1.0 if text and RE_PASCAL.match(text) else 0.0
    is_unspaced_op = 1.0 if text and text in {'=', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/='} and (line_text[max(0, start_col - 1)] != ' ' or line_text[min(len(line_text) - 1, end_col)] != ' ') else 0.0

    return [
        float(depth), float(len(node.children)), 1.0 if depth == 0 else 0.0, 1.0 if len(node.children) == 0 else 0.0,
        1.0 if node.type in control_triggers else 0.0, line_span, char_len,
        float(start_col), (start_col % 4) / 4.0, float(len(line_text)), 1.0 if line_text.endswith((' ', '\t')) else 0.0,
        1.0 if line_span > 1.0 else 0.0, is_snake, is_camel, is_pascal, float(text.count('_')) if text else 0.0,
        sum(c in 'aeiouAEIOU' for c in text) / max(1, len(text)) if text else 0.0, 1.0 if text and len(text) == 1 else 0.0,
        1.0 if 'type' in node.type else 0.0, 1.0 if 'string' in node.type else 0.0, 1.0 if ('subscript' in node.type or 'array' in node.type) else 0.0,
        1.0 if node.has_error else 0.0, shannon_entropy(text), len(set(text)) / len(text) if text else 0.0,
        math.log1p(token_to_rank.get(text, max_rank)) if text else 0.0, 1.0 if (node.type == 'identifier' and text and text.lower() in LAZY_IDENTIFIERS) else 0.0,
        is_unspaced_op, 1.0 if (('number' in node.type or 'integer' in node.type or 'float' in node.type) and text and (text not in MAGIC_NUM_EXCLUSIONS)) else 0.0,
        1.0 if node.type in DECISION_TYPES["python"] else 0.0, float(depth) / MAX_AST_DEPTH, float(sibling_idx) / max(1.0, float(num_siblings - 1)),
        1.0 if node.type in {'conditional_expression', 'ternary_expression'} else 0.0,
        1.0 if (text in {'+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '++', '--'} or 'compound_assignment' in node.type or 'augmented_assignment' in node.type) else 0.0,
        1.0 if (parent_type and parent_type == node.type) else 0.0
    ]


def _stylometry_cpp(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type):
    control_triggers = CONTROL_TRIGGERS["cpp"]
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    line_span = float(end_row - start_row + 1)
    char_len = float(node.end_byte - node.start_byte)
    line_text = source_lines[start_row] if start_row < len(source_lines) else ""

    is_snake = 1.0 if text and RE_SNAKE.match(text) else 0.0
    is_camel = 1.0 if text and RE_CAMEL.match(text) else 0.0
    is_pascal = 1.0 if text and RE_PASCAL.match(text) else 0.0
    is_unspaced_op = 1.0 if text and text in {'=', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/='} and (line_text[max(0, start_col - 1)] != ' ' or line_text[min(len(line_text) - 1, end_col)] != ' ') else 0.0

    return [
        float(depth), float(len(node.children)), 1.0 if depth == 0 else 0.0, 1.0 if len(node.children) == 0 else 0.0,
        1.0 if node.type in control_triggers else 0.0, line_span, char_len,
        float(start_col), (start_col % 4) / 4.0, float(len(line_text)), 1.0 if line_text.endswith((' ', '\t')) else 0.0,
        1.0 if line_span > 1.0 else 0.0, is_snake, is_camel, is_pascal, float(text.count('_')) if text else 0.0,
        sum(c in 'aeiouAEIOU' for c in text) / max(1, len(text)) if text else 0.0, 1.0 if text and len(text) == 1 else 0.0,
        1.0 if 'type' in node.type else 0.0, 1.0 if 'string' in node.type else 0.0, 1.0 if ('subscript' in node.type or 'array' in node.type) else 0.0,
        1.0 if node.has_error else 0.0, shannon_entropy(text), len(set(text)) / len(text) if text else 0.0,
        math.log1p(token_to_rank.get(text, max_rank)) if text else 0.0, 1.0 if (node.type == 'identifier' and text and text.lower() in LAZY_IDENTIFIERS) else 0.0,
        is_unspaced_op, 1.0 if (('number' in node.type and text and (text not in MAGIC_NUM_EXCLUSIONS))) else 0.0,
        1.0 if node.type in DECISION_TYPES["cpp"] else 0.0, float(depth) / MAX_AST_DEPTH, float(sibling_idx) / max(1.0, float(num_siblings - 1)),
        1.0 if node.type in {'conditional_expression', 'ternary_expression'} else 0.0,
        1.0 if (text in {'+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '++', '--'} or 'compound_assignment' in node.type) else 0.0,
        1.0 if (parent_type and parent_type == node.type) else 0.0
    ]


def _stylometry_java(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type):
    control_triggers = CONTROL_TRIGGERS["java"]
    start_row, start_col = node.start_point
    end_row, end_col = node.end_point
    line_text = source_lines[start_row] if start_row < len(source_lines) else ""

    is_snake = 1.0 if text and RE_SNAKE.match(text) else 0.0
    is_camel = 1.0 if text and RE_CAMEL.match(text) else 0.0
    is_pascal = 1.0 if text and RE_PASCAL.match(text) else 0.0
    is_unspaced_op = 1.0 if text and text in {'=', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/='} and (line_text[max(0, start_col - 1)] != ' ' or line_text[min(len(line_text) - 1, end_col)] != ' ') else 0.0

    return [
        float(depth), float(len(node.children)), 1.0 if depth == 0 else 0.0, 1.0 if len(node.children) == 0 else 0.0,
        1.0 if node.type in control_triggers else 0.0, float(end_row - start_row + 1), float(node.end_byte - node.start_byte),
        float(start_col), (start_col % 4) / 4.0, float(len(line_text)), 1.0 if line_text.endswith((' ', '\t')) else 0.0,
        1.0 if (end_row - start_row + 1) > 1.0 else 0.0, is_snake, is_camel, is_pascal, float(text.count('_')) if text else 0.0,
        sum(c in 'aeiouAEIOU' for c in text) / max(1, len(text)) if text else 0.0, 1.0 if text and len(text) == 1 else 0.0,
        1.0 if 'type' in node.type else 0.0, 1.0 if 'string' in node.type else 0.0, 1.0 if 'array_access' in node.type else 0.0,
        1.0 if node.has_error else 0.0, shannon_entropy(text), len(set(text)) / len(text) if text else 0.0,
        math.log1p(token_to_rank.get(text, max_rank)) if text else 0.0, 1.0 if node.type == 'identifier' and text and text.lower() in LAZY_IDENTIFIERS else 0.0,
        is_unspaced_op, 1.0 if 'literal' in node.type and text and text not in MAGIC_NUM_EXCLUSIONS else 0.0,
        1.0 if node.type in DECISION_TYPES["java"] else 0.0, float(depth) / MAX_AST_DEPTH, float(sibling_idx) / max(1.0, float(num_siblings - 1)),
        1.0 if node.type == 'ternary_expression' else 0.0, 1.0 if (text in {'+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '++', '--'} or 'assignment_expression' in node.type) else 0.0,
        1.0 if parent_type == node.type else 0.0
    ]


def extract_node_stylometry(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx=0, num_siblings=1, parent_type="", language="python"):
    if language == "java":
        return _stylometry_java(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type)
    elif language == "cpp":
        return _stylometry_cpp(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type)
    else:
        return _stylometry_python(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type)


# ---------------------------------------------------------------------------
# Macro features (16-d). Exact per-notebook bodies.
# ---------------------------------------------------------------------------

def _macro_python(source_code, source_lines, nodes_info):
    total_lines = max(1, len(source_lines))
    non_empty_lines = [l for l in source_lines if l.strip()]
    num_non_empty = max(1, len(non_empty_lines))

    snake_c = sum(1 for n in nodes_info if n['struct'][12] == 1.0)
    camel_c = sum(1 for n in nodes_info if n['struct'][13] == 1.0)
    pascal_c = sum(1 for n in nodes_info if n['struct'][14] == 1.0)
    total_c = snake_c + camel_c + pascal_c
    case_consistency = max(snake_c, camel_c, pascal_c) / float(max(1, total_c))

    indents = [len(l) - len(l.lstrip(' ')) for l in non_empty_lines]
    indent_step_mods = [ind % 4 for ind in indents]
    indent_step_variance = float(np.var(indent_step_mods)) if indent_step_mods else 0.0
    indent_variance = float(np.var(indents)) if indents else 0.0

    comment_lines = sum(1 for l in source_lines if l.strip().startswith(('#', '//', '/*', '*')))
    comment_to_code_ratio = float(comment_lines) / float(total_lines)
    tactical_comments = sum(1 for n in nodes_info if 'comment' in n['type'] and n.get('text') and RE_TACTICAL.search(n['text']))
    total_comments = sum(1 for n in nodes_info if 'comment' in n['type'])
    tactical_comment_ratio = float(tactical_comments) / max(1.0, float(total_comments))

    decision_depths = [n['f_depth'] for n in nodes_info if n['struct'][28] == 1.0]
    max_nesting_depth = float(max(decision_depths)) if decision_depths else 0.0
    avg_nesting_depth = float(np.mean(decision_depths)) if decision_depths else 0.0

    line_lengths = [len(l) for l in non_empty_lines]
    line_len_variance = float(np.var(line_lengths)) if line_lengths else 0.0
    long_line_ratio = sum(1 for l in line_lengths if l > 80) / float(num_non_empty)

    preproc_count = sum(1 for n in nodes_info if n['type'] in {'preproc_def', 'preproc_include', 'import_statement', 'import_from_statement'})
    macro_density = float(preproc_count) / float(total_lines)

    blank_intervals, cur_interval = [], 0
    for l in source_lines:
        if not l.strip():
            if cur_interval > 0:
                blank_intervals.append(cur_interval)
                cur_interval = 0
        else:
            cur_interval += 1
    if cur_interval > 0:
        blank_intervals.append(cur_interval)
    empty_line_entropy = float(shannon_entropy("".join(str(min(9, i)) for i in blank_intervals)))

    raw_bytes = bytes(source_code, "utf8")
    gzip_ratio = len(gzip.compress(raw_bytes)) / max(1.0, float(len(raw_bytes)))

    operators = [n['type_id'] for n in nodes_info if n['type'] not in TEXT_CAPTURE_TYPES["python"]]
    operands = [n.get('text') for n in nodes_info if n['type'] in TEXT_CAPTURE_TYPES["python"]]
    n_hal, eta_hal = len(operators) + len(operands), len(set(operators)) + len(set(operands))
    halstead_volume_log = math.log1p(n_hal * math.log2(max(1.0, float(eta_hal))))

    funcs = [n for n in nodes_info if n['type'] in {'function_definition', 'method_declaration'}]
    func_max_to_mean_ratio = float(max([f['line_span'] for f in funcs])) / max(1.0, float(np.mean([f['line_span'] for f in funcs]))) if funcs else 1.0
    param_count = sum(1 for n in nodes_info if n['type'] in {'parameter_declaration', 'parameter', 'identifier'} and n['struct'][33] == 0)
    avg_param_count = float(param_count) / max(1.0, float(len(funcs)))
    call_to_def_ratio = float(sum(1 for n in nodes_info if n['type'] in {'call_expression', 'call'})) / max(1.0, float(len(funcs)))

    return [
        case_consistency, indent_step_variance, indent_variance, comment_to_code_ratio, tactical_comment_ratio,
        max_nesting_depth, avg_nesting_depth, line_len_variance, long_line_ratio, macro_density, empty_line_entropy,
        gzip_ratio, halstead_volume_log, func_max_to_mean_ratio, avg_param_count, call_to_def_ratio
    ]


def _macro_cpp(source_code, source_lines, nodes_info):
    total_lines = max(1, len(source_lines))
    non_empty = [l for l in source_lines if l.strip()]
    funcs = [n for n in nodes_info if n['type'] == 'function_definition']
    operators = [n['type_id'] for n in nodes_info if n['type'] not in TEXT_CAPTURE_TYPES["cpp"]]
    operands = [n.get('text') for n in nodes_info if n['type'] in TEXT_CAPTURE_TYPES["cpp"]]

    blank_intervals, cur_interval = [], 0
    for l in source_lines:
        if not l.strip():
            if cur_interval > 0:
                blank_intervals.append(cur_interval)
                cur_interval = 0
        else:
            cur_interval += 1
    if cur_interval > 0:
        blank_intervals.append(cur_interval)

    return [
        max(sum(1 for n in nodes_info if n['struct'][12] == 1.0), sum(1 for n in nodes_info if n['struct'][13] == 1.0), sum(1 for n in nodes_info if n['struct'][14] == 1.0)) / max(1, sum(1 for n in nodes_info if n['struct'][12] == 1.0) + sum(1 for n in nodes_info if n['struct'][13] == 1.0) + sum(1 for n in nodes_info if n['struct'][14] == 1.0)),
        float(np.var([ind % 4 for ind in [len(l) - len(l.lstrip(' ')) for l in non_empty]])) if non_empty else 0.0,
        float(np.var([len(l) - len(l.lstrip(' ')) for l in non_empty])) if non_empty else 0.0,
        float(sum(1 for l in source_lines if l.strip().startswith(('//', '/*', '*')))) / float(total_lines),
        float(sum(1 for n in nodes_info if 'comment' in n['type'] and n.get('text') and RE_TACTICAL.search(n['text']))) / max(1.0, float(sum(1 for n in nodes_info if 'comment' in n['type']))),
        float(max([n['f_depth'] for n in nodes_info if n['struct'][28] == 1.0] + [0.0])),
        float(np.mean([n['f_depth'] for n in nodes_info if n['struct'][28] == 1.0] + [0.0])),
        float(np.var([len(l) for l in non_empty])) if non_empty else 0.0,
        sum(1 for l in non_empty if len(l) > 80) / max(1, len(non_empty)),
        float(sum(1 for n in nodes_info if n['type'] in {'preproc_def', 'preproc_include'})) / float(total_lines),
        shannon_entropy("".join(str(min(9, i)) for i in blank_intervals)),
        len(gzip.compress(bytes(source_code, "utf8"))) / max(1.0, float(len(source_code))),
        math.log1p((len(operators) + len(operands)) * math.log2(max(1.0, float(len(set(operators)) + len(set(operands)))))),
        float(max([f['line_span'] for f in funcs] + [1.0])) / max(1.0, float(np.mean([f['line_span'] for f in funcs] + [1.0]))),
        float(sum(1 for n in nodes_info if n['type'] in {'parameter_declaration', 'parameter'})) / max(1.0, float(len(funcs))),
        float(sum(1 for n in nodes_info if n['type'] == 'call_expression')) / max(1.0, float(len(funcs)))
    ]


def _macro_java(source_code, source_lines, nodes_info):
    non_empty = [l for l in source_lines if l.strip()]
    funcs = [n for n in nodes_info if n['type'] in {'method_declaration', 'constructor_declaration'}]
    operators = [n['type_id'] for n in nodes_info if n['type'] not in TEXT_CAPTURE_TYPES["java"]]
    operands = [n.get('text') for n in nodes_info if n['type'] in TEXT_CAPTURE_TYPES["java"]]

    return [
        max(sum(1 for n in nodes_info if n['struct'][12] == 1.0), sum(1 for n in nodes_info if n['struct'][13] == 1.0), sum(1 for n in nodes_info if n['struct'][14] == 1.0)) / max(1, sum(1 for n in nodes_info if n['struct'][12] == 1.0) + sum(1 for n in nodes_info if n['struct'][13] == 1.0) + sum(1 for n in nodes_info if n['struct'][14] == 1.0)),
        float(np.var([ind % 4 for ind in [len(l) - len(l.lstrip(' ')) for l in non_empty]])) if non_empty else 0.0,
        float(np.var([len(l) - len(l.lstrip(' ')) for l in non_empty])) if non_empty else 0.0,
        float(sum(1 for l in source_lines if l.strip().startswith(('//', '/*', '*')))) / max(1, len(source_lines)),
        float(sum(1 for n in nodes_info if 'comment' in n['type'] and n.get('text') and RE_TACTICAL.search(n['text']))) / max(1.0, float(sum(1 for n in nodes_info if 'comment' in n['type']))),
        float(max([n['f_depth'] for n in nodes_info if n['struct'][28] == 1.0] + [0.0])),
        float(np.mean([n['f_depth'] for n in nodes_info if n['struct'][28] == 1.0] + [0.0])),
        float(np.var([len(l) for l in non_empty])) if non_empty else 0.0,
        sum(1 for l in non_empty if len(l) > 80) / max(1, len(non_empty)),
        float(sum(1 for n in nodes_info if n['type'] in {'import_declaration', 'package_declaration'})) / max(1, len(source_lines)),
        shannon_entropy("".join(str(min(9, i)) for i in [len(chunk) for chunk in re.split(r'[^\n]', source_code) if '\n' in chunk])),
        len(gzip.compress(bytes(source_code, "utf8"))) / max(1.0, float(len(source_code))),
        math.log1p((len(operators) + len(operands)) * math.log2(max(1.0, float(len(set(operators)) + len(set(operands)))))),
        float(max([f['line_span'] for f in funcs] + [1.0])) / max(1.0, float(np.mean([f['line_span'] for f in funcs] + [1.0]))),
        float(sum(1 for n in nodes_info if n['type'] == 'formal_parameter')) / max(1.0, float(len(funcs))),
        float(sum(1 for n in nodes_info if n['type'] == 'method_invocation')) / max(1.0, float(len(funcs)))
    ]


def compute_macro_features(source_code, source_lines, nodes_info, language="python"):
    if language == "java":
        return _macro_java(source_code, source_lines, nodes_info)
    elif language == "cpp":
        return _macro_cpp(source_code, source_lines, nodes_info)
    else:
        return _macro_python(source_code, source_lines, nodes_info)


# ---------------------------------------------------------------------------
# Vocab / BPE fitting (shared skeleton, per-language desc preserved)
# ---------------------------------------------------------------------------

VOCAB_DESC = {
    "python": "Fitting Python Tokenizer & Node Vocab",
    "java": "Fitting Tokenizer & Node Vocabulary",
    "cpp": "Fitting C++ Tokenizer & Node Vocab",
}


def fit_vocab_and_tokenizer(train_clean_data, parser, language):
    counter, train_identifiers = Counter(), []
    capture = TEXT_CAPTURE_TYPES[language]
    for item in tqdm(train_clean_data, desc=VOCAB_DESC.get(language, "Fitting Tokenizer & Node Vocab")):
        code = item.get('text', item.get('code', ''))
        if not code:
            continue
        tree = parser.parse(bytes(code, "utf8"))
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.is_named or 'comment' in node.type:
                counter[node.type] += 1
                if node.type in capture:
                    train_identifiers.append(code[node.start_byte:node.end_byte])
            stack.extend(node.children)

    type_to_id = {t: i + 1 for i, t in enumerate([t for t, _ in counter.most_common(MAX_VOCAB_SIZE)])}
    vocab_size = len(type_to_id) + 1
    bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    bpe_tokenizer.train_from_iterator(train_identifiers, trainers.BpeTrainer(special_tokens=["[PAD]", "[UNK]"], vocab_size=BPE_VOCAB_SIZE))
    pad_id = bpe_tokenizer.token_to_id("[PAD]")
    token_freqs = Counter(train_identifiers)
    token_to_rank = {token: rank + 1 for rank, token in enumerate([k for k, v in token_freqs.most_common()])}
    max_rank = len(token_to_rank) + 1
    return type_to_id, vocab_size, bpe_tokenizer, pad_id, token_to_rank, max_rank


# ---------------------------------------------------------------------------
# Graph construction with language dispatch (outputs identical to notebooks)
# ---------------------------------------------------------------------------

class GraphContext:
    def __init__(self, language, parser, type_to_id, vocab_size, bpe_tokenizer, pad_id, token_to_rank, max_rank):
        self.language = language
        self.parser = parser
        self.type_to_id = type_to_id
        self.vocab_size = vocab_size
        self.bpe_tokenizer = bpe_tokenizer
        self.pad_id = pad_id
        self.token_to_rank = token_to_rank
        self.max_rank = max_rank
        self.means = None
        self.stds = None
        self.g_means = None
        self.g_stds = None


def build_optimized_ast_graph(source_code, label, ctx):
    language = ctx.language if isinstance(ctx, GraphContext) else ctx
    if isinstance(ctx, GraphContext):
        parser = ctx.parser
        type_to_id = ctx.type_to_id
        bpe_tokenizer = ctx.bpe_tokenizer
        pad_id = ctx.pad_id
        token_to_rank = ctx.token_to_rank
        max_rank = ctx.max_rank
    else:
        raise ValueError("ctx must be GraphContext")
    capture = TEXT_CAPTURE_TYPES[language]
    scopes = SCOPE_TYPES[language]

    if not source_code or not isinstance(source_code, str):
        return None
    try:
        tree = parser.parse(bytes(source_code, "utf8"))
    except Exception:
        return None

    source_lines, nodes, edges, edge_types, subwords_list, leaf_order = source_code.splitlines(), [], [], [], [], []
    ast_id_to_node_id, scope_stack, cfg_seqs, func_registry, calls = {}, [{}], [], {}, []

    def get_identifier_text(n):
        if n.type == 'identifier':
            return source_code[n.start_byte:n.end_byte]
        for c in n.children:
            res = get_identifier_text(c)
            if res:
                return res
        return None

    def get_parameter_ids(n, param_list):
        if language == "python":
            if n.type == 'identifier':
                param_list.append(n.id)
                return
            for c in n.children:
                get_parameter_ids(c, param_list)
        elif language == "cpp":
            if n.type == 'parameter_declaration':
                param_list.append(n.id)
            for c in n.children:
                get_parameter_ids(c, param_list)
        else:  # java
            if n.type == 'formal_parameter':
                param_list.append(n.id)
            for c in n.children:
                get_parameter_ids(c, param_list)

    def finalize_scope(scope_uses):
        for _text, uses in scope_uses.items():
            for i in range(len(uses) - 1):
                edges.extend([[uses[i], uses[i + 1]], [uses[i + 1], uses[i]]])
                edge_types.extend([4, 5])

    def traverse(node, depth, parent_id, scope_uses, sibling_idx=0, num_siblings=1, parent_type=""):
        node_id = len(nodes)
        ast_id_to_node_id[node.id] = node_id
        if language == "python":
            is_scope = node.type in scopes or node.type in {'block', 'module'}
        elif language == "cpp":
            is_scope = node.type in scopes or node.type == 'compound_statement'
        else:
            is_scope = node.type in scopes or node.type in {'block', 'program'}
        if is_scope:
            scope_stack.append({})

        text = source_code[node.start_byte:node.end_byte] if node.type in capture else None
        if language == "java":
            enc = bpe_tokenizer.encode(text).ids[:MAX_SUBWORDS] + [pad_id] * MAX_SUBWORDS if text else [pad_id] * MAX_SUBWORDS
            subwords_list.append(enc[:MAX_SUBWORDS])
        else:
            if node.type in capture and text:
                enc = bpe_tokenizer.encode(text).ids[:MAX_SUBWORDS]
                subwords_list.append(enc + [pad_id] * (MAX_SUBWORDS - len(enc)))
            else:
                subwords_list.append([pad_id] * MAX_SUBWORDS)

        nodes.append({
            'type_id': type_to_id.get(node.type, 0), 'type': node.type,
            'struct': extract_node_stylometry(node, text, depth, source_lines, token_to_rank, max_rank, sibling_idx, num_siblings, parent_type, language),
            'text': text, 'line_span': float(node.end_point[0] - node.start_point[0] + 1), 'f_depth': float(depth)
        })

        if node.type == 'identifier' and text:
            scope_uses.setdefault(text, []).append(node_id)
            if language == "python":
                is_def = (parent_type in {'assignment', 'augmented_assignment', 'parameters', 'typed_parameter', 'default_parameter', 'with_item', 'for_statement'} and sibling_idx == 0)
            elif language == "cpp":
                is_def = (parent_type in {'init_declarator', 'parameter_declaration', 'declaration'} or (parent_type == 'assignment_expression' and sibling_idx == 0))
            else:
                is_def = (parent_type in {'variable_declarator', 'formal_parameter', 'local_variable_declaration'} or (parent_type == 'assignment_expression' and sibling_idx == 0))
            if is_def:
                scope_stack[-1][text] = node_id
            else:
                for scope in reversed(scope_stack):
                    if text in scope and scope[text] != node_id:
                        edges.extend([[scope[text], node_id], [node_id, scope[text]]])
                        edge_types.extend([9, 10])
                        break

        if language == "python":
            if node.type == 'function_definition':
                name_node = node.child_by_field_name('name')
                fname = get_identifier_text(name_node) if name_node else None
                if name_node and fname:
                    params = []
                    param_node = node.child_by_field_name('parameters')
                    if param_node:
                        get_parameter_ids(param_node, params)
                    func_registry[fname] = {'def': node_id, 'params': params}
            elif node.type == 'call':
                func_node = node.child_by_field_name('function')
                fname = get_identifier_text(func_node) if func_node else None
                if func_node and fname:
                    args = []
                    args_node = node.child_by_field_name('arguments')
                    if args_node:
                        args = [c.id for c in args_node.children if c.is_named]
                    calls.append({'call_node': node_id, 'fname': fname, 'args': args})
            if node.type in {'block', 'module'}:
                named_children = [c for c in node.children if c.is_named]
                for i in range(len(named_children) - 1):
                    cfg_seqs.append((named_children[i].id, named_children[i + 1].id))
        elif language == "cpp":
            if node.type == 'function_definition':
                decl = node.child_by_field_name('declarator')
                fname = get_identifier_text(decl) if decl else None
                if decl and fname:
                    params = []
                    get_parameter_ids(node, params)
                    func_registry[fname] = {'def': node_id, 'params': params}
            elif node.type == 'call_expression':
                func_node = node.child_by_field_name('function')
                fname = get_identifier_text(func_node) if func_node else None
                if func_node and fname:
                    args = []
                    args_node = node.child_by_field_name('arguments')
                    if args_node:
                        args = [c.id for c in args_node.children if c.is_named]
                    calls.append({'call_node': node_id, 'fname': fname, 'args': args})
            if node.type == 'compound_statement':
                named_children = [c for c in node.children if c.is_named]
                for i in range(len(named_children) - 1):
                    cfg_seqs.append((named_children[i].id, named_children[i + 1].id))
        else:  # java
            if node.type in {'method_declaration', 'constructor_declaration'}:
                name_node = node.child_by_field_name('name')
                fname = get_identifier_text(name_node) if name_node else None
                if name_node and fname:
                    params = []
                    get_parameter_ids(node, params)
                    func_registry[fname] = {'def': node_id, 'params': params}
            elif node.type == 'method_invocation':
                name_node = node.child_by_field_name('name')
                fname = get_identifier_text(name_node) if name_node else None
                if name_node and fname:
                    args_node = node.child_by_field_name('arguments')
                    args = [c.id for c in args_node.children if c.is_named] if args_node else []
                    calls.append({'call_node': node_id, 'fname': fname, 'args': args})
            if node.type in {'block', 'program'}:
                named_children = [c for c in node.children if c.is_named]
                for i in range(len(named_children) - 1):
                    cfg_seqs.append((named_children[i].id, named_children[i + 1].id))

        if parent_id is not None:
            edges.extend([[parent_id, node_id], [node_id, parent_id]])
            edge_types.extend([0, 1])
        if len(node.children) == 0:
            leaf_order.append(node_id)

        if language == "python":
            opens_scope = node.type in scopes
            child_scope_uses = {} if opens_scope else scope_uses
            name_node = node.child_by_field_name('name') if opens_scope else None
        elif language == "cpp":
            opens_scope = node.type in scopes
            child_scope_uses = {} if opens_scope else scope_uses
            name_node = node.child_by_field_name('declarator') if opens_scope else None
        else:
            _is_block_scope = node.type in scopes or node.type in {'block', 'program'}
            child_scope_uses = {} if _is_block_scope else scope_uses
            name_node = node.child_by_field_name('name') if _is_block_scope else None
            opens_scope = None  # java uses _is_block_scope below

        named_children = [c for c in node.children if c.is_named or 'comment' in c.type]
        prev_sibling_id = None
        for c_idx, child in enumerate(named_children):
            if len(nodes) >= MAX_AST_NODES or depth + 1 > MAX_AST_DEPTH:
                continue
            child_id = traverse(child, depth + 1, node_id, scope_uses if child is name_node else child_scope_uses, c_idx, len(named_children), node.type)
            if prev_sibling_id is not None:
                edges.extend([[prev_sibling_id, child_id], [child_id, prev_sibling_id]])
                edge_types.extend([2, 3])
            prev_sibling_id = child_id

        if language in ("python", "cpp"):
            if opens_scope:
                finalize_scope(child_scope_uses)
        else:
            if node.type in scopes or node.type in {'block', 'program'}:
                for _text, uses in child_scope_uses.items():
                    for i in range(len(uses) - 1):
                        edges.extend([[uses[i], uses[i + 1]], [uses[i + 1], uses[i]]])
                        edge_types.extend([4, 5])
                scope_stack.pop()
                return node_id
        if is_scope:
            scope_stack.pop()
        return node_id

    try:
        traverse(tree.root_node, 0, None, {})
    except Exception:
        return None

    if language == "java":
        for src, dst in cfg_seqs:
            if src in ast_id_to_node_id and dst in ast_id_to_node_id:
                edges.extend([[ast_id_to_node_id[src], ast_id_to_node_id[dst]]])
                edge_types.append(11)
    else:
        for src, dst in cfg_seqs:
            if src in ast_id_to_node_id and dst in ast_id_to_node_id:
                edges.append([ast_id_to_node_id[src], ast_id_to_node_id[dst]])
                edge_types.append(11)

    for c in calls:
        if c['fname'] in func_registry:
            fdef = func_registry[c['fname']]
            edges.extend([[c['call_node'], fdef['def']], [fdef['def'], c['call_node']]])
            edge_types.extend([12, 13])
            for arg_id, param_id in zip(c['args'], fdef['params']):
                if arg_id in ast_id_to_node_id and param_id in ast_id_to_node_id:
                    edges.extend([[ast_id_to_node_id[arg_id], ast_id_to_node_id[param_id]], [ast_id_to_node_id[param_id], ast_id_to_node_id[arg_id]]])
                    edge_types.extend([14, 15])

    for i in range(len(leaf_order) - 1):
        edges.extend([[leaf_order[i], leaf_order[i + 1]], [leaf_order[i + 1], leaf_order[i]]])
        edge_types.extend([6, 7])

    if not nodes or not edges:
        return None

    global_stats = torch.tensor([compute_macro_features(source_code, source_lines, nodes, language)], dtype=torch.float)
    virtual_node_id = len(nodes)
    virtual_struct = [0.0] * 34
    virtual_struct[2] = 1.0
    nodes.append({'type_id': 0, 'type': 'virtual_node', 'struct': virtual_struct, 'text': None, 'line_span': 0.0, 'f_depth': 0.0})
    subwords_list.append([pad_id] * MAX_SUBWORDS)

    for i in range(len(nodes) - 1):
        if language == "java":
            edges.extend([[i, virtual_node_id]])
        else:
            edges.append([i, virtual_node_id])
        edge_types.append(8)

    return Data(
        x_type=torch.tensor([n['type_id'] for n in nodes], dtype=torch.long),
        x_struct=torch.tensor([n['struct'] for n in nodes], dtype=torch.float),
        x_subwords=torch.tensor(subwords_list, dtype=torch.long),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_types, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.float),
        num_nodes=len(nodes),
        virtual_idx=virtual_node_id,
        global_stats=global_stats
    )


def process_split(split, desc, ctx):
    out = []
    for item in tqdm(split, desc=desc):
        g = build_optimized_ast_graph(item.get('text', item.get('code', '')), item.get('label', item.get('target', 0)), ctx)
        if g is not None:
            out.append(g)
    return out


def fit_normalization(train_clean_graphs, ctx):
    _all_structs = torch.cat([g.x_struct for g in train_clean_graphs])
    means, stds = _all_structs[:, CONTINUOUS_IDX].mean(dim=0), _all_structs[:, CONTINUOUS_IDX].std(dim=0).clamp(min=1e-6)
    _all_global = torch.cat([g.global_stats for g in train_clean_graphs], dim=0)
    g_means, g_stds = _all_global.mean(dim=0), _all_global.std(dim=0).clamp(min=1e-6)
    ctx.means, ctx.stds, ctx.g_means, ctx.g_stds = means, stds, g_means, g_stds
    return means, stds, g_means, g_stds


def apply_normalization(graphs, ctx):
    for g in graphs:
        g.x_struct[:, CONTINUOUS_IDX] = (g.x_struct[:, CONTINUOUS_IDX] - ctx.means) / ctx.stds
        g.global_stats = (g.global_stats - ctx.g_means) / ctx.g_stds
    return graphs
