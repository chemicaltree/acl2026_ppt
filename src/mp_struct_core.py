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
        use_head_diversity=False,
        use_complex_args=False,
        seed=42
    ):
        self.rng = random.Random(seed)
        self.k_struct = k_struct
        self.k_dep = k_dep
        self.use_head_diversity = use_head_diversity
        self.use_complex_args = use_complex_args

        self.total_k = k_struct + k_dep
        self.bracket_vocab_size = self.total_k * 2

        self.DEP_MOVE = k_struct + 0
        self.DEP_AGR_A = k_struct + 1
        self.DEP_AGR_B = k_struct + 2
        self.DEP_SEL = k_struct + 3

        self.HEAD_CP_ID = self.bracket_vocab_size + 0
        self.HEAD_TP_ID = self.bracket_vocab_size + 1
        self.HEAD_VP_ID = self.bracket_vocab_size + 2

        self.LEAF_ID = self.bracket_vocab_size + 3

    def _open(self, id): return id
    def _close(self, id): return id + self.total_k

    def _wrap_struct(self, content_ids):
        s_id = 0
        return [self._open(s_id)] + content_ids + [self._close(s_id)]

    def _gen_local_pair(self):
        p1 = self._wrap_struct([self._open(self.DEP_SEL)])
        p2 = self._wrap_struct([self._close(self.DEP_SEL)])
        return p1 + p2

    def _get_head_token(self, layer_type):
        if not self.use_head_diversity:
            return []

        if layer_type == 'CP':
            return [self.HEAD_CP_ID]
        elif layer_type == 'TP':
            return [self.HEAD_TP_ID]
        elif layer_type == 'VP':
            return [self.HEAD_VP_ID]
        return []

    def get_vocab(self):
        vocab = {}

        for x in range(self.bracket_vocab_size):
            total_k = self.total_k
            is_open = (x < total_k)
            base_id = x if is_open else x - total_k

            if base_id < self.k_struct:
                sym = f"[{base_id}" if is_open else f"]{base_id}"
            else:
                sym = f"({base_id}" if is_open else f"){base_id}"
            vocab[sym] = x

        vocab["HEAD_CP"] = self.HEAD_CP_ID
        vocab["HEAD_TP"] = self.HEAD_TP_ID
        vocab["HEAD_VP"] = self.HEAD_VP_ID

        vocab["LEAF"] = self.LEAF_ID

        return vocab

    def generate_tree_sequence(self):
        has_move = self.rng.random() < 0.5
        agree_type = self.DEP_AGR_A if self.rng.random() < 0.5 else self.DEP_AGR_B

        args = []

        if has_move:
            subj_slot = [self._close(self.DEP_MOVE)]
        else:
            subj_slot = []

        if self.use_complex_args:
            rand_val = self.rng.random()
            if rand_val < 0.33:
                pass
            elif rand_val < 0.66:
                args.append(self._gen_local_pair())
            else:
                args.append(self._gen_local_pair())
                args.append(self._gen_local_pair())
        else:
            args.append(self._gen_local_pair())

        head_v_tokens = self._get_head_token('VP')
        head_v = self._wrap_struct(head_v_tokens)

        vp_inner = []
        elements = [head_v, self._wrap_struct(subj_slot)] + [self._wrap_struct(a) for a in args]
        self.rng.shuffle(elements)

        for e in elements: vp_inner += e

        vp_block = self._wrap_struct(vp_inner)

        head_t_tokens = self._get_head_token('TP')
        head_t_content = head_t_tokens + [self._close(agree_type)]
        head_t = self._wrap_struct(head_t_content)

        subj_content = self._gen_local_pair()
        subj_block = self._wrap_struct([self._open(agree_type)] + subj_content)

        if has_move:
            spec_tp = []
        else:
            spec_tp = subj_block

        tp_inner = []
        if spec_tp: tp_inner += spec_tp
        else: tp_inner += self._wrap_struct([])

        tp_inner += head_t
        tp_inner += vp_block

        tp_block = self._wrap_struct(tp_inner)

        head_c_tokens = self._get_head_token('CP')
        head_c_content = head_c_tokens
        if has_move:
            head_c_content += [self._open(self.DEP_MOVE)]

        head_c = self._wrap_struct(head_c_content)

        spec_cp = []
        if has_move:
            spec_cp = subj_block

        cp_inner = []
        if spec_cp: cp_inner += self._wrap_struct(spec_cp)
        else: cp_inner += self._wrap_struct([])

        cp_inner += head_c
        cp_inner += tp_block

        cp_block = self._wrap_struct(cp_inner)

        return cp_block


def generate(
    out_dir="./data/mp_struct_core",
    n=100000,
    k_struct=1,
    k_dep=4,
    use_head_diversity=True,
    use_complex_args=False,
    length=1024,
    seed=42
):
    print(f"Generating to {out_dir} (N={n})...")
    print(f" - Head Diversity: {use_head_diversity}")
    print(f" - Complex Args: {use_complex_args}")

    os.makedirs(out_dir, exist_ok=True)

    generator = EnhancedConstrainedGenerator(
        k_struct, k_dep, use_head_diversity, use_complex_args, seed
    )

    id_path = os.path.join(out_dir, "mp_struct_core_ids.txt")
    tok_path = os.path.join(out_dir, "mp_struct_core_tokens.txt")
    vocab_path = os.path.join(out_dir, "vocab.json")

    vocab = generator.get_vocab()
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)

    with open(id_path, "w") as f_ids, open(tok_path, "w") as f_tok:
        for _ in trange(n):
            line_ids = []
            while len(line_ids) < length:
                sent_ids = generator.generate_tree_sequence()
                line_ids.extend(sent_ids)

            line_ids = line_ids[:length]

            tokens = []
            for x in line_ids:
                if x < generator.bracket_vocab_size:
                    total_k = generator.total_k
                    is_open = (x < total_k)
                    base_id = x if is_open else x - total_k

                    if base_id < k_struct:
                        sym = f"[{base_id}" if is_open else f"]{base_id}"
                    else:
                        sym = f"({base_id}" if is_open else f"){base_id}"
                elif x == generator.HEAD_CP_ID: sym = "HEAD_CP"
                elif x == generator.HEAD_TP_ID: sym = "HEAD_TP"
                elif x == generator.HEAD_VP_ID: sym = "HEAD_VP"
                else: sym = "UNK"

                tokens.append(sym)

            f_ids.write(" ".join(map(str, line_ids)) + "\n")
            f_tok.write(" ".join(tokens) + "\n")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    fire.Fire({"generate": generate})
