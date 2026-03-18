import os, sys, time, json, math, random, uuid, re
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# =================== Configuración ===================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBED_MODEL_LOCAL = "all-MiniLM-L6-v2"

# Configuración del proceso
BP_VARIANTS_PER_CALL = int(os.getenv("BP_VARIANTS_PER_CALL", "5")) # Aumentado ligeramente
MIN_JACCARD_FOR_LABEL = float(os.getenv("MIN_JACCARD_FOR_LABEL", "0.20")) # Bajado un poco para ser mas flexible
MAX_JACCARD_FOR_DIVERSITY = float(os.getenv("MAX_JACCARD_FOR_DIVERSITY", "0.95"))
TARGET_AUG_PER_CLASS = int(os.getenv("TARGET_AUG_PER_CLASS", "200")) 

INPUT_JSONL  = os.getenv("INPUT_JSONL",  "merged_masked_unionv2_new/merged_masked_unionv2_new_train_low_regime.jsonl")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "merged_masked_unionv2_new/merged_masked_unionv2_new_train_low_regime_qwen.jsonl")

USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "1") not in ("0","false","False")
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.90")) # Subido un poco para permitir más variantes
MAX_NEG_EXAMPLES = 3
RAG_NEIGHBORS_PER_CAT = 64

SEED = 42
random.seed(SEED)

# =================== Globales ===================
local_tokenizer = None
local_model = None
local_embedder = None
openai_client = None

# =================== Inicialización ===================
def init_models():
    global local_tokenizer, local_model, local_embedder, openai_client

    if LLM_PROVIDER == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                with open("API_KEY", "r") as f: api_key = f.read().strip()
            except: pass
        openai_client = OpenAI(api_key=api_key)
        print(f"[INIT] OpenAI client initialized.")

    elif LLM_PROVIDER == "local":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[INIT] Loading local model: {LOCAL_MODEL_NAME}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        attn_impl = "eager" 

        try:
            local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME, trust_remote_code=True)
            if local_tokenizer.pad_token is None:
                local_tokenizer.pad_token = local_tokenizer.eos_token

            local_model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_NAME, 
                torch_dtype=torch.float32, 
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation=attn_impl
            )
            
        except Exception as e:
            print(f"[WARN] Error loading model: {e}")
            sys.exit(1)

        if USE_EMBEDDINGS:
            print(f"[INIT] Loading local embedder: {EMBED_MODEL_LOCAL}")
            from sentence_transformers import SentenceTransformer
            local_embedder = SentenceTransformer(EMBED_MODEL_LOCAL, device="cpu") # Embedder en CPU para ahorrar VRAM

# =================== Gestión de Archivos ===================
import csv

def read_input_auto(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path): return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def initialize_output_file(rows: List[Dict[str, Any]]):
    Path(os.path.dirname(OUTPUT_JSONL) or ".").mkdir(parents=True, exist_ok=True)
    if not os.path.exists(OUTPUT_JSONL):
        print(f"[IO] Creando {OUTPUT_JSONL}...")
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        print(f"[IO] Añadiendo a {OUTPUT_JSONL} existente.")

def append_to_output(row: Dict[str, Any]):
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# =================== Utils & Robustez ===================
STRICT_TOKEN = re.compile(r"\[(SLUR|TARGET):([A-Z0-9_]+)\]")

def _strip_fences(s: str) -> str:
    # Elimina bloques de código markdown
    s = re.sub(r"```(?:\w+)?", "", s)
    s = s.replace("```", "")
    return s.strip()

def repair_and_validate_tokens(orig_tokens: List[str], generated_text: str) -> Tuple[bool, str]:
    """
    Intenta arreglar tokens mal formados (ej: [ SLUR : ID ]) y verifica si están todos.
    Retorna (True, texto_arreglado) si tiene éxito.
    """
    repaired_text = generated_text
    
    # 1. Normalización agresiva de tokens en el texto generado
    # Busca patrones tipo [ ALGO : ID ] con espacios y los compacta
    def replacer(match):
        return f"[{match.group(1)}:{match.group(2)}]"
    
    repaired_text = re.sub(r"\[\s*(SLUR|TARGET)\s*:\s*([A-Z0-9_]+)\s*\]", replacer, repaired_text)
    
    # 2. Verificación
    missing = []
    for tok in orig_tokens:
        if tok not in repaired_text:
            missing.append(tok)
            
    if missing:
        return False, repaired_text # Aún faltan tokens
    
    return True, repaired_text

def normalize_text_for_jaccard(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", str(s).lower()).strip()
    return [s[i:i+3] for i in range(max(1, len(s)-2))]

def jaccard_sim(a: str, b: str) -> float:
    A = set(normalize_text_for_jaccard(a))
    B = set(normalize_text_for_jaccard(b))
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

# =================== Memoria Robusta ===================
class MemoryStore:
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        self.by_cat_texts: Dict[str, List[str]] = defaultdict(list)
        self.by_cat_embs:  Dict[str, List[List[float]]] = defaultdict(list)

    def _embed_batch(self, texts: List[str]):
        if not texts or not self.use_embeddings or local_embedder is None: return []
        try:
            # Force CPU execution for embeddings to avoid VRAM fragmentation
            embeddings = local_embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings.tolist()
        except: return []

    def _cos_sim(self, u, v) -> float:
        try:
            dot = sum(x*y for x,y in zip(u,v))
            norm_u = math.sqrt(sum(x*x for x in u))
            norm_v = math.sqrt(sum(y*y for y in v))
            if norm_u == 0 or norm_v == 0: return 0.0
            return dot / (norm_u * norm_v)
        except: return 0.0

    def add(self, cat: str, texts: List[str]):
        valid_texts = [str(t) for t in texts if t and str(t).strip()]
        if not valid_texts: return
        self.by_cat_texts[cat].extend(valid_texts)
        if self.use_embeddings:
            embs = self._embed_batch(valid_texts)
            if len(embs) == len(valid_texts):
                self.by_cat_embs[cat].extend(embs)

    def nearest(self, cat: str, query: str, k: int = 5) -> List[Tuple[str, float]]:
        corpus = self.by_cat_texts.get(cat, [])
        if not corpus: return []
        if self.use_embeddings and self.by_cat_embs.get(cat):
            q_embs = self._embed_batch([query])
            if not q_embs: return [] 
            q_emb = q_embs[0]
            limit = min(len(corpus), len(self.by_cat_embs[cat]))
            sims = [(corpus[i], self._cos_sim(q_emb, self.by_cat_embs[cat][i])) for i in range(limit)]
        else:
            sims = [(t, jaccard_sim(query, t)) for t in corpus]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def too_similar(self, cat: str, query: str, threshold: float) -> bool:
        if not self.by_cat_texts.get(cat): return False
        try:
            neighbors = self.nearest(cat, query, k=1)
            return any(sim >= threshold for _, sim in neighbors)
        except: return False

# =================== Generación LLM ===================
# PROMPT SIMPLIFICADO: No pedimos JSON, solo lista con guiones
SYSTEM_PROMPT = (
    "You are a data augmentation assistant. "
    "Rewrite the toxic tweet provided by the user, keeping the same meaning and toxicity.\n"
    "RULES:\n"
    "1. You MUST retain the bracketed tokens (e.g., [SLUR:ID], [TARGET:ID]) EXACTLY as they appear.\n"
    "2. Do not explain. Do not translate tokens.\n"
    "3. Output ONLY the new variants, one per line, starting with '- '."
)

def call_llm(system_prompt: str, user_prompt: str, max_new_tokens: int = 256) -> str:
    # Aumentamos temperatura ligeramente para evitar bucles repetitivos
    temp = 1.1 
    top_p = 0.95
    
    try:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text = local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        model_inputs = local_tokenizer([text], return_tensors="pt")
        input_ids = model_inputs.input_ids
        
        # Limpieza de tokens fuera de rango
        max_allowed = local_model.get_input_embeddings().weight.shape[0] - 1
        if (input_ids >= max_allowed).any():
            input_ids[input_ids >= max_allowed] = local_tokenizer.eos_token_id
        
        model_inputs = model_inputs.to(local_model.device)

        generated_ids = local_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2, # Penalización más alta para evitar loops
            pad_token_id=local_tokenizer.eos_token_id
        )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return local_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    except Exception as e:
        print(f"[LLM ERR] {str(e)[:100]}")
        return ""

def generate_variants(masked_text: str, cat: str, lang: str, mem: MemoryStore, n: int) -> List[str]:
    negs = mem.nearest(cat, masked_text, k=MAX_NEG_EXAMPLES)
    neg_txt = "\n".join([f"Avoid: {t}" for t, _ in negs]) if negs else ""
    
    user_prompt = (
        f"Original: {masked_text}\n"
        f"{neg_txt}\n"
        f"Generate {n} diverse rewrites (same language '{lang}'):"
    )
    
    raw_out = call_llm(SYSTEM_PROMPT, user_prompt)
    
    # Parseo de lista con guiones
    variants = []
    for line in raw_out.split('\n'):
        clean = _strip_fences(line).strip()
        if clean.startswith("- "):
            clean = clean[2:].strip()
        elif clean.startswith("* "):
            clean = clean[2:].strip()
        elif re.match(r"^\d+\.", clean):
            clean = re.sub(r"^\d+\.\s*", "", clean)
            
        if clean and len(clean) > 10:
            variants.append(clean)
            
    return variants

# =================== Main ===================
def main():
    init_models()
    print(f"--- MODE: {LLM_PROVIDER} ---")
    print(f"--- TARGET: {TARGET_AUG_PER_CLASS} per class ---")
    
    rows = read_input_auto(INPUT_JSONL)
    # Filtrar filas válidas que tengan tokens
    valid_rows = [r for r in rows if isinstance(r.get("text_masked"), str) and STRICT_TOKEN.search(r["text_masked"])]
    
    initialize_output_file(rows)

    by_cat = defaultdict(list)
    for r in valid_rows:
        by_cat[r.get("predicted_hate_category", "unknown")].append(r)

    memory = MemoryStore(use_embeddings=USE_EMBEDDINGS)
    # Carga inicial de memoria
    for cat, items in by_cat.items():
        base_texts = [it["text_masked"] for it in items[:RAG_NEIGHBORS_PER_CAT]]
        memory.add(cat, base_texts)

    # Ordenar categorías por cantidad de datos (menos datos primero, o como prefieras)
    # Aquí procesamos todas.
    sorted_cats = list(by_cat.keys())
    
    total_augmented = 0
    
    # Calcular total esperado solo para barra de progreso
    total_needed = len(sorted_cats) * TARGET_AUG_PER_CLASS
    pbar = tqdm(total=total_needed, desc="Augmenting", unit="tweets")

    for cat in sorted_cats:
        items = by_cat[cat]
        if not items: continue
        
        current_aug_count = 0
        target = TARGET_AUG_PER_CLASS
        
        # Mezclamos los items originales para variedad
        random.shuffle(items)
        
        # Índice circular para recorrer los items hasta completar el objetivo
        idx = 0
        fails_consecutive_global = 0
        
        while current_aug_count < target:
            if fails_consecutive_global > 20:
                print(f"[WARN] Demasiados fallos consecutivos en categoría {cat}. Saltando...")
                break
                
            row = items[idx % len(items)]
            idx += 1
            
            orig_text = str(row["text_masked"])
            # Extraer tokens requeridos
            orig_tokens = [m.group(0) for m in STRICT_TOKEN.finditer(orig_text)]
            
            # Generar
            variants = generate_variants(orig_text, cat, row.get("lang","en"), memory, n=BP_VARIANTS_PER_CALL)
            
            valid_batch_count = 0
            for v in variants:
                # 1. Reparar y validar tokens
                ok_tokens, v_fixed = repair_and_validate_tokens(orig_tokens, v)
                if not ok_tokens:
                    # Debug silencioso: print(f"SKIP (Tokens): {v}") 
                    continue

                # 2. Validar Jaccard (Similitud con original)
                jsim = jaccard_sim(orig_text, v_fixed)
                if jsim < MIN_JACCARD_FOR_LABEL:
                    # print(f"SKIP (Low Sim {jsim:.2f}): {v_fixed}")
                    continue
                if jsim > MAX_JACCARD_FOR_DIVERSITY:
                    continue
                
                # 3. Validar Memoria (Novedad)
                if memory.too_similar(cat, v_fixed, SIM_THRESHOLD):
                    # print(f"SKIP (Memory): {v_fixed}")
                    continue
                
                # ÉXITO
                memory.add(cat, [v_fixed])
                new_row = dict(row)
                new_row["id"] = f"{row.get('id')}_aug_{uuid.uuid4().hex[:6]}"
                new_row["text_masked"] = v_fixed
                new_row["augmented"] = True
                new_row["model"] = LOCAL_MODEL_NAME
                
                append_to_output(new_row)
                
                current_aug_count += 1
                total_augmented += 1
                valid_batch_count += 1
                pbar.update(1)
                
                if current_aug_count >= target:
                    break
            
            if valid_batch_count == 0:
                fails_consecutive_global += 1
            else:
                fails_consecutive_global = 0 # Reset si conseguimos al menos uno

    pbar.close()
    print(f"\n[DONE] Proceso completado. Filas generadas: {total_augmented}")

if __name__ == "__main__":
    main()