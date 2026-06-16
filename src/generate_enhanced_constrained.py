import os
import random
import json
import fire
from tqdm import trange

class EnhancedConstrainedGenerator:
    def __init__(
        self, 
        k_struct=1, 
        k_dep=4, 
        use_head_diversity=False, # 機能範疇（Head）の区別を入れるか
        use_complex_args=False,   # 項構造（自動詞/他動詞）の複雑さを入れるか
        seed=42
    ):
        self.rng = random.Random(seed)
        self.k_struct = k_struct
        self.k_dep = k_dep
        self.use_head_diversity = use_head_diversity
        self.use_complex_args = use_complex_args
        
        # ID Offsets
        # Brackets: 0 .. total_k*2 - 1
        self.total_k = k_struct + k_dep
        self.bracket_vocab_size = self.total_k * 2
        
        # Dependency Roles
        self.DEP_MOVE = k_struct + 0
        self.DEP_AGR_A = k_struct + 1
        self.DEP_AGR_B = k_struct + 2
        self.DEP_SEL = k_struct + 3

        # Head Token IDs (Leaf Terminals)
        # 括弧IDの後ろに追加する
        self.HEAD_CP_ID = self.bracket_vocab_size + 0
        self.HEAD_TP_ID = self.bracket_vocab_size + 1
        self.HEAD_VP_ID = self.bracket_vocab_size + 2
        
        # Filler/Empty (Leafなしの場合は使わないが、念のため)
        self.LEAF_ID = self.bracket_vocab_size + 3

    def _open(self, id): return id
    def _close(self, id): return id + self.total_k

    def _wrap_struct(self, content_ids):
        """Wrap content in Structure Bracket [0 ... ]0"""
        s_id = 0 
        return [self._open(s_id)] + content_ids + [self._close(s_id)]

    def _gen_local_pair(self):
        """Generate Local Selection Pair: [0 (4 ]0 [0 )4 ]0"""
        p1 = self._wrap_struct([self._open(self.DEP_SEL)])
        p2 = self._wrap_struct([self._close(self.DEP_SEL)])
        return p1 + p2

    def _get_head_token(self, layer_type):
        """
        Return head token ID if diversity is enabled.
        layer_type: 'CP', 'TP', 'VP'
        """
        if not self.use_head_diversity:
            return [] # No token
            
        if layer_type == 'CP':
            return [self.HEAD_CP_ID]
        elif layer_type == 'TP':
            return [self.HEAD_TP_ID]
        elif layer_type == 'VP':
            return [self.HEAD_VP_ID]
        return []

    def get_vocab(self):
        """Generate vocabulary dictionary (Token -> ID)"""
        vocab = {}
        
        # 1. Brackets
        for x in range(self.bracket_vocab_size):
            total_k = self.total_k
            is_open = (x < total_k)
            base_id = x if is_open else x - total_k
            
            if base_id < self.k_struct:
                sym = f"[{base_id}" if is_open else f"]{base_id}"
            else:
                sym = f"({base_id}" if is_open else f"){base_id}"
            vocab[sym] = x
            
        # 2. Head Tokens (only if diversity is enabled, or always include to keep ID space consistent)
        # To match the generation logic, we include them if they correspond to valid IDs.
        # Even if use_head_diversity is False, we might want to know what ID corresponds to what symbol conceptually,
        # but for strict dataset usage, we only need symbols that appear.
        # Here we add them regardless to ensure vocab file covers the ID space defined in __init__.
        vocab["HEAD_CP"] = self.HEAD_CP_ID
        vocab["HEAD_TP"] = self.HEAD_TP_ID
        vocab["HEAD_VP"] = self.HEAD_VP_ID
        
        # 3. Leaf/Filler
        vocab["LEAF"] = self.LEAF_ID
        
        return vocab

    def generate_tree_sequence(self):
        """
        Generates sequence with optional Diversity and Complexity.
        """
        
        # 1. Decide Global Parameters
        has_move = self.rng.random() < 0.5
        agree_type = self.DEP_AGR_A if self.rng.random() < 0.5 else self.DEP_AGR_B
        
        # --- Bottom Layer (VP) ---
        
        # Argument Structure Complexity
        args = []
        
        # 1. Subject Slot
        if has_move:
            subj_slot = [self._close(self.DEP_MOVE)] # Trace
        else:
            subj_slot = [] # Base position (Leafless)
        
        # 2. Decide Valency (Arguments)
        if self.use_complex_args:
            rand_val = self.rng.random()
            if rand_val < 0.33:
                # Intransitive
                pass 
            elif rand_val < 0.66:
                # Transitive
                args.append(self._gen_local_pair()) # Obj1
            else:
                # Ditransitive
                args.append(self._gen_local_pair()) # Obj1
                args.append(self._gen_local_pair()) # Obj2
        else:
            # Default fixed: Transitive
            args.append(self._gen_local_pair()) # Obj

        # VP Assembly
        head_v_tokens = self._get_head_token('VP')
        head_v = self._wrap_struct(head_v_tokens)
        
        vp_inner = []
        elements = [head_v, self._wrap_struct(subj_slot)] + [self._wrap_struct(a) for a in args]
        self.rng.shuffle(elements)
        
        for e in elements: vp_inner += e
        
        vp_block = self._wrap_struct(vp_inner)


        # --- Middle Layer (TP) ---
        
        # Head T
        head_t_tokens = self._get_head_token('TP')
        head_t_content = head_t_tokens + [self._close(agree_type)]
        head_t = self._wrap_struct(head_t_content)
        
        # Spec TP
        subj_content = self._gen_local_pair()
        subj_block = self._wrap_struct([self._open(agree_type)] + subj_content)
        
        if has_move:
            spec_tp = [] 
        else:
            spec_tp = subj_block
            
        # TP Assembly
        tp_inner = []
        if spec_tp: tp_inner += spec_tp
        else: tp_inner += self._wrap_struct([])
        
        tp_inner += head_t
        tp_inner += vp_block
        
        tp_block = self._wrap_struct(tp_inner)


        # --- Top Layer (CP) ---
        
        # Head C
        head_c_tokens = self._get_head_token('CP')
        head_c_content = head_c_tokens
        if has_move:
            head_c_content += [self._open(self.DEP_MOVE)]
        
        head_c = self._wrap_struct(head_c_content)
        
        # Spec CP
        spec_cp = []
        if has_move:
            spec_cp = subj_block

        # CP Assembly
        cp_inner = []
        if spec_cp: cp_inner += self._wrap_struct(spec_cp)
        else: cp_inner += self._wrap_struct([])
        
        cp_inner += head_c
        cp_inner += tp_block
        
        cp_block = self._wrap_struct(cp_inner)
        
        return cp_block


def generate_dataset(
    out_dir="./data/enhanced_constrained",
    n=100000,
    k_struct=1,
    k_dep=4,
    use_head_diversity=False,
    use_complex_args=False,
    length=1024,
    seed=42
):
    """
    Generate Enhanced Structure-Constrained Dataset.
    """
    
    print(f"Generating to {out_dir} (N={n})...")
    print(f" - Head Diversity: {use_head_diversity}")
    print(f" - Complex Args: {use_complex_args}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    generator = EnhancedConstrainedGenerator(
        k_struct, k_dep, use_head_diversity, use_complex_args, seed
    )
    
    # Define file paths
    id_path = os.path.join(out_dir, "hybrid_ids.txt")
    tok_path = os.path.join(out_dir, "hybrid_tokens.txt")
    vocab_path = os.path.join(out_dir, "vocab.json")
    
    # Save Vocab
    vocab = generator.get_vocab()
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    
    # Generate Data
    with open(id_path, "w") as f_ids, open(tok_path, "w") as f_tok:
        for _ in trange(n):
            line_ids = []
            while len(line_ids) < length:
                sent_ids = generator.generate_tree_sequence()
                line_ids.extend(sent_ids)
            
            line_ids = line_ids[:length]
            
            # Token conversion
            tokens = []
            for x in line_ids:
                # Brackets
                if x < generator.bracket_vocab_size:
                    total_k = generator.total_k
                    is_open = (x < total_k)
                    base_id = x if is_open else x - total_k
                    
                    if base_id < k_struct:
                        sym = f"[{base_id}" if is_open else f"]{base_id}"
                    else:
                        sym = f"({base_id}" if is_open else f"){base_id}"
                # Head Tokens
                elif x == generator.HEAD_CP_ID: sym = "HEAD_CP"
                elif x == generator.HEAD_TP_ID: sym = "HEAD_TP"
                elif x == generator.HEAD_VP_ID: sym = "HEAD_VP"
                else: sym = "UNK"
                    
                tokens.append(sym)
            
            f_ids.write(" ".join(map(str, line_ids)) + "\n")
            f_tok.write(" ".join(tokens) + "\n")

    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    fire.Fire(generate_dataset)