import re
import math

# SymPy関連 (インポートエラー対策)
try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify, N
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# ==========================================
# A. 解答抽出・正規化
# ==========================================
def normalize_text(text):
    """
    表記ゆれを吸収するための正規化関数
    """
    if not text: return ""
    text = str(text).strip()
    # 1. LaTeXコマンドの簡易正規化
    text = text.replace(r"\dfrac", r"\frac")
    text = text.replace(r"\tfrac", r"\frac")
    text = text.replace(r"\left", "").replace(r"\right", "")
    # 2. 空白の削除 (数式において空白は意味を持たないことが多い)
    text = text.replace(" ", "")
    return text

def extract_answer_content(text):
    """
    テキストから \boxed{...} の中身を抽出する。
    見つからない場合は None を返す。
    """
    if not text: return None
    # 最後の \boxed{...} を抽出 (貪欲マッチを避けるため非貪欲に)
    # ネストした括弧には対応していない簡易正規表現ですが、NuminaMathなら概ねOK
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if not matches: return None
    
    content = matches[-1]
    # ボックスの中にさらにボックスがある場合の対策 (稀にある)
    # 例: \boxed{\boxed{42}} -> 42
    nested = re.findall(r"\\boxed\{(.*?)\}", content)
    if nested:
        return nested[-1].strip()
        
    return content.strip()

# ==========================================
# B. 正誤判定ロジック (Robust)
# ==========================================
def is_number(s):
    """文字列が数値変換可能か判定"""
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False

def parse_digits(s):
    """安全な数値変換 (カンマ除去など)"""
    return float(str(s).replace(",", ""))

def robust_float_check(pred, gold):
    """
    数値的な一致を確認する (evalを使わない安全版)
    分数、パーセント、小数に対応
    """
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    # 1. 完全一致 (正規化後)
    if pred == gold:
        return True

    # 2. パーセント処理 (例: 50% == 0.5)
    if pred.endswith("%") and not gold.endswith("%"):
        try:
            return abs(parse_digits(pred[:-1])/100 - parse_digits(gold)) < 1e-6
        except: pass
    if gold.endswith("%") and not pred.endswith("%"):
        try:
            return abs(parse_digits(pred) - parse_digits(gold[:-1])/100) < 1e-6
        except: pass

    # 3. 分数処理 (例: 1/2 == 0.5, \frac{1}{2} == 0.5)
    # 簡易的な分数パーサ
    def parse_fraction(s):
        # \frac{A}{B} 形式
        frac_match = re.match(r"\\frac\{([\d\.]+)\}\{([\d\.]+)\}", s)
        if frac_match:
            try:
                return float(frac_match.group(1)) / float(frac_match.group(2))
            except: pass
        # A/B 形式
        if "/" in s:
            parts = s.split("/")
            if len(parts) == 2 and is_number(parts[0]) and is_number(parts[1]):
                try:
                    return float(parts[0]) / float(parts[1])
                except: pass
        # 通常の数値
        if is_number(s):
            return parse_digits(s)
        return None

    val_pred = parse_fraction(pred)
    val_gold = parse_fraction(gold)

    if val_pred is not None and val_gold is not None:
        return abs(val_pred - val_gold) < 1e-6

    return False

def check_equivalence(pred_str, gold_str):
    """
    SymPyを用いた厳密な等価性判定 + 数値判定
    """
    if not pred_str or not gold_str: return False
    
    # まずは文字列としての正規化比較
    if normalize_text(pred_str) == normalize_text(gold_str):
        return True
    
    # SymPyによる解析 (最も信頼性が高い)
    if SYMPY_AVAILABLE:
        try:
            # タイムアウト機構がないため、複雑すぎる式はここに入れないよう注意
            # latex2sympyは \text{} などに弱いため、簡易除去してから渡す
            clean_pred = pred_str.replace(r"\text", "").replace("$", "")
            clean_gold = gold_str.replace(r"\text", "").replace("$", "")
            
            sym_pred = latex2sympy(clean_pred)
            sym_gold = latex2sympy(clean_gold)
            
            # 数式的な引き算をして0になるか ( simplify(a-b) == 0 )
            diff = simplify(sym_pred - sym_gold)
            if diff == 0:
                return True
            
            # 数値的な評価 (evalf) での比較
            # (例えば sin^2 + cos^2 - 1 = 0 が simplify で落ちない場合など)
            if abs(N(diff)) < 1e-6:
                return True
                
        except Exception:
            # パースエラーや計算エラーは無視して、次のチェックへ
            pass
            
    # SymPyでダメだった場合のフォールバック（数値チェック）
    return robust_float_check(pred_str, gold_str)

def is_correct(pred_text, gold_text):
    """
    生成テキスト全体と正解テキストを受け取り、正誤を判定するラッパー関数
    """
    pred_ans = extract_answer_content(pred_text)
    gold_ans = extract_answer_content(gold_text)
    return check_equivalence(pred_ans, gold_ans)


# ==========================================
# C. ステップ分割・統合 (Adaptive Splitting)
# ==========================================
def reduce_step_count(steps, target_max=15, min_chars=50):
    if len(steps) <= target_max: return steps
    
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: 
            buffer = step
            continue
        
        if len(step) < min_chars or len(buffer) < min_chars:
            buffer += "\n" + step
        else:
            merged_steps.append(buffer)
            buffer = step
            
    if buffer: merged_steps.append(buffer)
    
    while len(merged_steps) > target_max:
        new_merged = []
        for i in range(0, len(merged_steps), 2):
            if i + 1 < len(merged_steps):
                new_merged.append(merged_steps[i] + "\n" + merged_steps[i+1])
            else:
                new_merged.append(merged_steps[i])
        merged_steps = new_merged
        if len(merged_steps) <= 1: break
        
    return merged_steps

def split_text_into_steps(path_text):
    """
    解答テキストを論理的なステップに分割する。
    """
    steps = [s.strip() for s in re.split(r'\n\s*\n', path_text) if s.strip()]
    
    if len(steps) < 3:
        alt_steps = [s.strip() for s in path_text.split('\n') if s.strip()]
        if len(alt_steps) >= 3: 
            steps = alt_steps
            
    if len(steps) > 15:
        steps = reduce_step_count(steps, target_max=15, min_chars=50)
        
    return steps

# ==========================================
# D. その他
# ==========================================
def is_suitable_for_prm(solution_text: str) -> bool:
    ans = extract_answer_content(solution_text)
    if not ans: return False
    if len(ans) > 100 or len(ans) == 0: return False
    if "\\\\" in ans or "\n" in ans: return False
    if ans.count("=") > 1: return False
    if "\\begin" in ans: return False
    return True

def build_prompt(tokenizer, problem_text: str) -> str:
    system_prompt = "Please reason step by step and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem_text}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)