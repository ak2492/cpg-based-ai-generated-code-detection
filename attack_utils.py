"""Attack primitives + evaluation helpers — exact notebook logic, combined.

Python/C++ regex primitives share one helper where outputs are identical
(radical rename differs only by RESERVED set). Java tree-sitter primitives
are preserved verbatim. Test-time generators
(generate_python/java/cpp_attack_samples) are kept per-language exact.
Eval helpers: execute_model_eval (java, no-cost) + execute_model_eval_with_cost
(shared) + print helpers + distribution-shift audit.
"""
import random
import re
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from language_configs import get_parser, get_reserved

try:
    import psutil as _psutil
    _psutil_proc = _psutil.Process()
except ImportError:  # pragma: no cover - Kaggle always has psutil
    _psutil = None
    _psutil_proc = None


def current_rss_mb():
    """Current process RSS in MB (0.0 if psutil unavailable)."""
    if _psutil_proc is None:
        return 0.0
    return _psutil_proc.memory_info().rss / (1024 * 1024)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared radical rename (python/cpp identical modulo RESERVED) — output-preserving
# ---------------------------------------------------------------------------

def _radical_rename(code_str, reserved):
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_str)
    var_map = {}
    v_idx = 1
    for t in tokens:
        if t not in reserved and t not in var_map:
            var_map[t] = f"v_{v_idx}"
            v_idx += 1
    return re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lambda m: var_map.get(m.group(0), m.group(0)), code_str)


def radical_variable_renaming_python(code_str):
    from language_configs import PYTHON_RESERVED
    return _radical_rename(code_str, PYTHON_RESERVED)


def radical_variable_renaming_cpp(code_str):
    from language_configs import CPP_RESERVED
    return _radical_rename(code_str, CPP_RESERVED)


def radical_variable_renaming(code_str, language):
    return _radical_rename(code_str, get_reserved(language))


# ---------------------------------------------------------------------------
# Training-time obfuscation (python / cpp exact)
# ---------------------------------------------------------------------------

def apply_source_obfuscation_python(code_str):
    code_str = re.sub(r'#.*', '', code_str)
    code_str = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', code_str)
    code_str = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code_str)
    code_str = radical_variable_renaming_python(code_str)
    lines, new_lines = code_str.split('\n'), []
    for line in lines:
        stripped = line.strip()
        if stripped:
            # In Python, preserve leading indent to maintain AST syntactic validity, randomize trailing ws
            leading_ws = line[:len(line) - len(line.lstrip())]
            trailing_ws = " " * random.choice([0, 1, 2, 4])
            new_lines.append(leading_ws + stripped + trailing_ws)
        if random.random() < 0.05:
            new_lines.append("")
    return "\n".join(new_lines)


def apply_source_obfuscation_cpp(code_str):
    code_str = re.sub(r'//.*', '', code_str)
    code_str = re.sub(r'/\*.*?\*/', '', code_str, flags=re.DOTALL)
    code_str = radical_variable_renaming_cpp(code_str)
    lines, new_lines = code_str.split('\n'), []
    for line in lines:
        stripped = line.strip()
        if stripped:
            new_lines.append((" " * random.choice([0, 2, 4, 8])) + stripped)
        if random.random() < 0.05:
            new_lines.append("")
    return "\n".join(new_lines)


# ---------------------------------------------------------------------------
# Java primitives (verbatim from java notebook Cell 1)
# ---------------------------------------------------------------------------

def _java_parser(parser=None):
    if parser is not None:
        return parser
    p, _ = get_parser("java")
    return p


def strip_comments_basic(code_str, parser=None):
    parser = _java_parser(parser)
    try:
        tree = parser.parse(bytes(code_str, "utf8"))
        comment_nodes = []

        def find_comments(node):
            if 'comment' in node.type:
                comment_nodes.append(node)
            for child in node.children:
                find_comments(child)
        find_comments(tree.root_node)
        comment_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_str, "utf8")
        for node in comment_nodes:
            del code_bytes[node.start_byte:node.end_byte]
        return code_bytes.decode("utf8", errors="ignore")
    except Exception:
        return code_str


def meaning_preserving_rename_basic(code_str, parser=None):
    from language_configs import JAVA_RESERVED
    parser = _java_parser(parser)
    try:
        tree = parser.parse(bytes(code_str, "utf8"))
        id_types = {'identifier', 'type_identifier'}
        allowed_parents = {'variable_declarator', 'formal_parameter', 'catch_formal_parameter', 'spread_parameter', 'field_declaration', 'enhanced_for_statement', 'resource', 'method_declaration', 'class_declaration', 'constructor_declaration'}
        target_names = set()
        stack = [tree.root_node]
        all_ids = []
        while stack:
            n = stack.pop()
            if n.type in id_types:
                all_ids.append(n)
                pt = n.parent.type if n.parent else ""
                if pt in allowed_parents:
                    name = code_str[n.start_byte:n.end_byte]
                    if name and name not in JAVA_RESERVED and not name.startswith("__"):
                        target_names.add(name)
            stack.extend(n.children)

        if not target_names:
            return code_str
        var_map = {n: f"v_{i+1}" for i, n in enumerate(sorted(target_names))}
        all_ids.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_str, "utf8")
        for n in all_ids:
            name = code_str[n.start_byte:n.end_byte]
            if name in var_map:
                code_bytes[n.start_byte:n.end_byte] = bytes(var_map[name], "utf8")
        return code_bytes.decode("utf8", errors="ignore")
    except Exception:
        return code_str


def apply_statistical_attack_basic_java(code_str):
    rng = random.Random()
    lines = code_str.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            indent = " " * rng.choice([0, 1, 2, 4])
            trailing = (" " * rng.choice([0, 0, 1, 1, 2]) if rng.random() < 0.5 else "")
            new_lines.append(indent + stripped + trailing)
            if rng.random() < 0.05:
                new_lines.append("")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def normalize_naming_style_java(code_str, parser=None):
    from language_configs import JAVA_RESERVED
    parser = _java_parser(parser)
    try:
        tree = parser.parse(bytes(code_str, "utf8"))
        stack = [tree.root_node]
        all_ids = []
        while stack:
            n = stack.pop()
            if n.type == 'identifier':
                all_ids.append(n)
            stack.extend(n.children)

        unique_names = {}
        for node in all_ids:
            name = code_str[node.start_byte:node.end_byte]
            if not name or name in JAVA_RESERVED:
                continue
            if name not in unique_names:
                new_name = name
                if re.search(r'[A-Z]', name) and not name.isupper():
                    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                    new_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
                elif name.isupper() and len(name) > 1:
                    new_name = name.lower()
                cleaned = re.sub(r'\d+', '', new_name)
                unique_names[name] = cleaned if (cleaned and cleaned != '_') else name.lower()

        if not unique_names:
            return code_str
        all_ids.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_str, "utf8")
        for node in all_ids:
            name = code_str[node.start_byte:node.end_byte]
            if name in unique_names and unique_names[name] != name:
                code_bytes[node.start_byte:node.end_byte] = bytes(unique_names[name], "utf8")
        return code_bytes.decode("utf8", errors="ignore")
    except Exception:
        return code_str


def normalize_layout_java(code_str):
    lines = code_str.split("\n")
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip())
        indent_chars = line[:leading]
        indent_level = indent_chars.count('\t') + (indent_chars.count(' ') // 4)
        normalized_indent = "    " * indent_level
        content = line.strip()
        content = re.sub(r'\s*(==|!=|<=|>=|<<|>>|&&|\|\||[=+\-*/<>!&|%^])\s*', r' \1 ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        new_lines.append(normalized_indent + content)
    return "\n".join(new_lines)


def meaning_preserving_rename_enhanced_java(code_str, parser=None):
    from language_configs import JAVA_RESERVED
    parser = _java_parser(parser)
    try:
        tree = parser.parse(bytes(code_str, "utf8"))
        id_types = {'identifier', 'type_identifier'}
        allowed_parents = {'variable_declarator', 'formal_parameter', 'catch_formal_parameter', 'spread_parameter', 'field_declaration', 'enhanced_for_statement', 'resource', 'method_declaration', 'class_declaration', 'constructor_declaration'}

        target_names = set()
        stack = [tree.root_node]
        all_nodes = []
        while stack:
            n = stack.pop()
            all_nodes.append(n)
            if n.type in id_types:
                pt = n.parent.type if n.parent else ""
                if pt in allowed_parents:
                    name = code_str[n.start_byte:n.end_byte]
                    if name and name not in JAVA_RESERVED and not name.startswith("__"):
                        target_names.add(name)
            stack.extend(n.children)

        var_map = {n: f"v_{i+1}" for i, n in enumerate(sorted(target_names))}

        id_nodes = [n for n in all_nodes if n.type in id_types]
        id_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_str, "utf8")
        for n in id_nodes:
            name = code_str[n.start_byte:n.end_byte]
            if name in var_map:
                code_bytes[n.start_byte:n.end_byte] = bytes(var_map[name], "utf8")
        code_str = code_bytes.decode("utf8", errors="ignore")

        tree = parser.parse(bytes(code_str, "utf8"))
        stack = [tree.root_node]
        string_nodes = []
        while stack:
            n = stack.pop()
            if n.type in {'string_literal', 'string'}:
                if not n.parent or n.parent.type != "expression_statement":
                    string_nodes.append(n)
            stack.extend(n.children)

        string_nodes.sort(key=lambda n: n.start_byte, reverse=True)
        code_bytes = bytearray(code_str, "utf8")
        for n in string_nodes:
            orig = code_str[n.start_byte:n.end_byte]
            if orig.startswith('"'):
                code_bytes[n.start_byte:n.end_byte] = bytes('"s"', "utf8")
        code_str = code_bytes.decode("utf8", errors="ignore")

        code_str = re.sub(r'System\.out\.println\s*\([^)]*\)', 'System.out.println("output")', code_str)
        return code_str
    except Exception:
        return code_str


def apply_statistical_attack_enhanced_java(code_str):
    lines = code_str.split("\n")
    new_lines = []
    for line in lines:
        if line.strip():
            indent = " " * random.choice([0, 1, 3, 5, 7])
            trailing = " " * random.randint(1, 4)
            new_lines.append(indent + line.strip() + trailing)
            if random.random() < 0.15:
                new_lines.append("")
    return "\n".join(new_lines)


def apply_full_attack_basic_java(code_str, parser=None):
    c = strip_comments_basic(code_str, parser)
    c = meaning_preserving_rename_basic(c, parser)
    c = apply_statistical_attack_basic_java(c)
    return c


def apply_full_attack_enhanced_java(code_str, parser=None):
    c = strip_comments_basic(code_str, parser)
    c = normalize_naming_style_java(c, parser)
    c = normalize_layout_java(c)
    c = meaning_preserving_rename_enhanced_java(c, parser)
    c = apply_statistical_attack_enhanced_java(c)
    return c


# ---------------------------------------------------------------------------
# Test-time generators (exact notebook Cell 5 logic per language)
# ---------------------------------------------------------------------------

def generate_python_attack_samples(dataset_split, attack_type, mode="enhanced"):
    attack_samples = []
    attack_all = (mode == "basic")
    for row in dataset_split:
        c = row.get('code') or row.get('text') or ''
        l = int(row.get('label', row.get('target', 0)))
        if not c:
            continue

        # BASE PAPER PROTOCOL: Only machine-generated instances are perturbed
        if attack_all or l == 1:
            c_mod = c
            if attack_type in ["auth", "full"]:
                c_mod = re.sub(r'#.*', '', c_mod)
                c_mod = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', c_mod)
                c_mod = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', c_mod)
            if attack_type in ["sem", "full"]:
                c_mod = radical_variable_renaming_python(c_mod)
            if attack_type in ["stat", "full"]:
                lines = c_mod.split('\n')
                new_lines = []
                for line in lines:
                    if line.strip():
                        leading_ws = line[:len(line) - len(line.lstrip())]
                        trailing_ws = " " * random.randint(1, 4)
                        new_lines.append(leading_ws + line.strip() + trailing_ws)
                        if random.random() < 0.10:
                            new_lines.append("")
                c_mod = "\n".join(new_lines)
            attack_samples.append({'code': c_mod, 'label': l})
        else:
            attack_samples.append({'code': c, 'label': l})
    return attack_samples


def generate_cpp_attack_samples(dataset_split, attack_type, mode="enhanced"):
    attack_samples = []
    attack_all = (mode == "basic")
    for row in dataset_split:
        c = row.get('code') or row.get('text') or ''
        l = int(row.get('label', row.get('target', 0)))
        if not c:
            continue

        # BASE PAPER PROTOCOL: Only machine-generated instances are perturbed
        if attack_all or l == 1:
            c_mod = c
            if attack_type in ["auth", "full"]:
                c_mod = re.sub(r'//.*', '', c_mod)
                c_mod = re.sub(r'/\*.*?\*/', '', c_mod, flags=re.DOTALL)
            if attack_type in ["sem", "full"]:
                c_mod = radical_variable_renaming_cpp(c_mod)
            if attack_type in ["stat", "full"]:
                lines = c_mod.split('\n')
                new_lines = []
                for line in lines:
                    if line.strip():
                        indent = " " * random.choice([0, 1, 3, 5, 7])
                        trailing = " " * random.randint(1, 4)
                        new_lines.append(indent + line.strip() + trailing)
                        if random.random() < 0.10:
                            new_lines.append("")
                c_mod = "\n".join(new_lines)
            attack_samples.append({'code': c_mod, 'label': l})
        else:
            attack_samples.append({'code': c, 'label': l})
    return attack_samples


def generate_java_attack_samples(dataset_split, attack_type, mode="enhanced", parser=None):
    attack_samples = []
    attack_all = (mode == "basic")
    for row in dataset_split:
        c = row.get('code') or row.get('text') or ''
        l = int(row.get('label', row.get('target', 0)))
        if not c:
            continue

        if attack_all or l == 1:
            c_mod = c
            if attack_type in ["auth", "full"]:
                if mode == "basic":
                    c_mod = strip_comments_basic(c_mod, parser)
                else:
                    c_mod = strip_comments_basic(c_mod, parser)
                    c_mod = normalize_naming_style_java(c_mod, parser)
                    c_mod = normalize_layout_java(c_mod)
            if attack_type in ["sem", "full"]:
                if mode == "basic":
                    c_mod = meaning_preserving_rename_basic(c_mod, parser)
                else:
                    c_mod = meaning_preserving_rename_enhanced_java(c_mod, parser)
            if attack_type in ["stat", "full"]:
                if mode == "basic":
                    c_mod = apply_statistical_attack_basic_java(c_mod)
                else:
                    c_mod = apply_statistical_attack_enhanced_java(c_mod)
            attack_samples.append({'code': c_mod, 'label': l})
        else:
            attack_samples.append({'code': c, 'label': l})
    return attack_samples


def generate_attack_samples(dataset_split, attack_type, mode="enhanced", language="python", parser=None):
    if language == "python":
        return generate_python_attack_samples(dataset_split, attack_type, mode)
    elif language == "cpp":
        return generate_cpp_attack_samples(dataset_split, attack_type, mode)
    else:
        return generate_java_attack_samples(dataset_split, attack_type, mode, parser)


ATTACK_SUITES = [
    ("Clean Test", "clean", "clean"),
    ("Auth (Basic)", "auth", "basic"),
    ("Auth (Enhanced)", "auth", "enhanced"),
    ("Stat (Basic)", "stat", "basic"),
    ("Stat (Enhanced)", "stat", "enhanced"),
    ("Sem (Basic)", "sem", "basic"),
    ("Sem (Enhanced)", "sem", "enhanced"),
    ("Full (Basic)", "full", "basic"),
    ("Full (Enhanced)", "full", "enhanced"),
]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def execute_model_eval(eval_model, loader, device, threshold=0.50):
    eval_model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs.extend(torch.sigmoid(eval_model(batch)).cpu().numpy().flatten())
            labels.extend(batch.y.cpu().numpy().flatten())

    probs, labels = np.array(probs), np.array(labels)
    if len(labels) == 0:
        return None
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'Acc': accuracy_score(labels, preds),
        'Prec': precision_score(labels, preds, zero_division=0),
        'Rec': recall_score(labels, preds, zero_division=0),
        'F1': f1_score(labels, preds, zero_division=0),
        'ROC': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        'FPR': fp / max(1, fp + tn),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'N': len(labels)
    }


def execute_model_eval_with_cost(eval_model, loader, device, threshold=0.50):
    eval_model.eval()
    probs, labels = [], []
    rss_before = current_rss_mb()
    t_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs.extend(torch.sigmoid(eval_model(batch)).cpu().numpy().flatten())
            labels.extend(batch.y.cpu().numpy().flatten())

    inf_duration = time.perf_counter() - t_start
    peak_ram_mb = max(rss_before, current_rss_mb())
    probs, labels = np.array(probs), np.array(labels)
    if len(labels) == 0:
        return None
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    latency_ms = (inf_duration / len(labels)) * 1000.0
    throughput = len(labels) / max(1e-6, inf_duration)

    return {
        'Acc': accuracy_score(labels, preds),
        'Prec': precision_score(labels, preds, zero_division=0),
        'Rec': recall_score(labels, preds, zero_division=0),
        'F1': f1_score(labels, preds, zero_division=0),
        'ROC': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        'FPR': fp / max(1, fp + tn),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'N': len(labels),
        'Latency_ms': latency_ms, 'Throughput': throughput,
        'PeakRAM_MB': peak_ram_mb,
    }


def print_detailed_metrics(model_name, res):
    print(f"{model_name}:")
    print(f"  Metrics -> Acc: {res['Acc']:.4f} | Prec: {res['Prec']:.4f} | Rec: {res['Rec']:.4f} | F1: {res['F1']:.4f} | ROC: {res['ROC']:.4f} | FPR: {res['FPR']:.4f}")
    print(f"  ConfMat -> TN: {res['TN']:<5} FP: {res['FP']:<5} FN: {res['FN']:<5} TP: {res['TP']:<5}")


def print_detailed_metrics_with_cost(model_name, res):
    print(f"{model_name}:")
    print(f"  Metrics -> Acc: {res['Acc']:.4f} | Prec: {res['Prec']:.4f} | Rec: {res['Rec']:.4f} | F1: {res['F1']:.4f} | ROC: {res['ROC']:.4f} | FPR: {res['FPR']:.4f}")
    print(f"  ConfMat -> TN: {res['TN']:<5} FP: {res['FP']:<5} FN: {res['FN']:<5} TP: {res['TP']:<5}")
    print(f"  Cost    -> Latency: {res['Latency_ms']:.2f} ms/graph | Throughput: {res['Throughput']:.2f} graphs/sec | PeakRAM: {res.get('PeakRAM_MB', 0.0):.2f} MB")


def evaluate_distribution_shift(train_graphs, ood_graphs, feature_names=None):
    import numpy as np
    import scipy.stats as stats
    from language_configs import MACRO_NAMES
    print("\n" + "=" * 70)
    print("DISTRIBUTIONAL SHIFT ANALYSIS (OOD QUANTITATIVE AUDIT)")
    print("=" * 70)

    train_stats = np.array([g.global_stats.squeeze().cpu().numpy() for g in train_graphs])
    ood_stats = np.array([g.global_stats.squeeze().cpu().numpy() for g in ood_graphs])

    num_features = train_stats.shape[1]
    if feature_names is None:
        feature_names = MACRO_NAMES if num_features == len(MACRO_NAMES) else [f"Macro_Feature_{i}" for i in range(num_features)]

    significant_shifts = 0
    print(f"{'Feature Name':<28} | {'Wasserstein Dist':<18} | {'KS-Test p-val':<14} | {'Status'}")
    print("-" * 75)

    for i in range(num_features):
        f_train = train_stats[:, i]
        f_ood = ood_stats[:, i]

        w_dist = stats.wasserstein_distance(f_train, f_ood)
        ks_stat, p_val = stats.ks_2samp(f_train, f_ood)

        is_shifted = p_val < 0.001
        if is_shifted:
            significant_shifts += 1
            status = "OOD SHIFT"
        else:
            status = "SIMILAR"

        print(f"{feature_names[i]:<28} | {w_dist:<18.4f} | {p_val:<14.2e} | {status}")

    shift_percentage = (significant_shifts / num_features) * 100
    print("-" * 75)
    print(f"Total Features Evaluated: {num_features}")
    print(f"Features Showing Significant Shift: {significant_shifts} ({shift_percentage:.1f}%)")

    if shift_percentage >= 60.0:
        print("[VERDICT] Statistically significant Out-Of-Distribution (OOD) shift confirmed.")
    else:
        print("[VERDICT] Target data is largely in-distribution relative to the training set.")
    print("=" * 70)
