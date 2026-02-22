#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Subtask 1 Trainer (Content-Debiased Version)

Key strategies to reduce Content Effect (TCE):
1. Template Abstraction: Replace entities with abstract symbols (A, B, C)
2. Adversarial Debiasing: GRL to make encoder plausibility-invariant
3. Quadrant-balanced Sampling: Equal weight to all 4 validity×plausibility combos
4. Plausibility-aware SCL: Same validity + different plausibility = positive pairs
5. Focal Loss: Higher weight for content-confusing cases (invalid+plausible, valid+implausible)
"""

import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import json
import math
import random
import argparse
import copy
import re
import statistics
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import Function

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# =============================================================================
# Utils
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_seeds(seeds_str: str):
    if seeds_str is None:
        return None
    s = str(seeds_str).strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def normalize_text(t: str) -> str:
    return " ".join(t.strip().split())


def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_distribution(samples, name="Data"):
    valid_count = sum(1 for s in samples if s.get("validity", True))
    invalid_count = len(samples) - valid_count
    print(f"\n{name}: {len(samples)} samples")
    print(f"  Valid: {valid_count} ({100*valid_count/len(samples):.1f}%)")
    print(f"  Invalid: {invalid_count} ({100*invalid_count/len(samples):.1f}%)")


def print_quadrant_distribution(samples, name="Data"):
    """Print distribution across all 4 quadrants"""
    quadrants = {"VP": 0, "VI": 0, "IP": 0, "II": 0, "unknown": 0}
    for s in samples:
        v = s.get("validity", None)
        p = s.get("plausibility", None)
        if v is None or p is None:
            quadrants["unknown"] += 1
        elif v and p:
            quadrants["VP"] += 1
        elif v and not p:
            quadrants["VI"] += 1
        elif not v and p:
            quadrants["IP"] += 1
        else:
            quadrants["II"] += 1
    
    total = len(samples)
    print(f"\n{name}: {total} samples (4-quadrant distribution)")
    print(f"  Valid+Plausible (VP):     {quadrants['VP']:4d} ({100*quadrants['VP']/total:.1f}%)")
    print(f"  Valid+Implausible (VI):   {quadrants['VI']:4d} ({100*quadrants['VI']/total:.1f}%)")
    print(f"  Invalid+Plausible (IP):   {quadrants['IP']:4d} ({100*quadrants['IP']/total:.1f}%) ← hard")
    print(f"  Invalid+Implausible (II): {quadrants['II']:4d} ({100*quadrants['II']/total:.1f}%)")
    if quadrants["unknown"] > 0:
        print(f"  Unknown plausibility:     {quadrants['unknown']:4d}")


def stratified_split(samples, train_ratio=0.85, seed=42):
    """Stratified split by validity (and plausibility if available)"""
    random.seed(seed)
    
    # Group by quadrant
    quadrants = {"VP": [], "VI": [], "IP": [], "II": [], "unknown": []}
    for s in samples:
        v = s.get("validity", None)
        p = s.get("plausibility", None)
        if v is None or p is None:
            quadrants["unknown"].append(s)
        elif v and p:
            quadrants["VP"].append(s)
        elif v and not p:
            quadrants["VI"].append(s)
        elif not v and p:
            quadrants["IP"].append(s)
        else:
            quadrants["II"].append(s)
    
    train_samples = []
    val_samples = []
    
    for key, group in quadrants.items():
        random.shuffle(group)
        n_train = int(len(group) * train_ratio)
        train_samples.extend(group[:n_train])
        val_samples.extend(group[n_train:])
    
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    return train_samples, val_samples


# =============================================================================
# Data Augmentation (Critical for reducing Content Effect)
# =============================================================================
class SyllogismAugmenter:
    """
    Augmentation strategies:
    1. Premise swapping (doesn't change validity)
    2. Quantifier synonym replacement
    3. Entity substitution (key for debiasing!)
    4. Conclusion marker variation
    """
    
    def __init__(
        self, 
        swap_premises_prob=0.3, 
        quantifier_replace_prob=0.3,
        entity_replace_prob=0.5,  # Higher prob for debiasing
        conclusion_marker_prob=0.3,
        seed=42
    ):
        random.seed(seed)
        self.swap_premises_prob = swap_premises_prob
        self.quantifier_replace_prob = quantifier_replace_prob
        self.entity_replace_prob = entity_replace_prob
        self.conclusion_marker_prob = conclusion_marker_prob
        
        # Quantifier synonyms (preserve logical meaning)
        self.quantifier_map = {
            "all": ["every", "each", "any"],
            "every": ["all", "each", "any"],
            "some": ["certain", "a few", "several"],
            "no": ["not any", "not a single", "none of the"],
        }
        
        # Conclusion markers
        self.conclusion_markers = [
            "therefore", "thus", "hence", "consequently", "so",
            "it follows that", "from this", "this means that",
            "we can conclude that", "it must be that"
        ]
        
        # Abstract entities for substitution (plausibility-neutral)
        self.abstract_entities = [
            ("alpha", "alphas"), ("beta", "betas"), ("gamma", "gammas"),
            ("zeta", "zetas"), ("omega", "omegas"), ("delta", "deltas"),
            ("thing", "things"), ("object", "objects"), ("entity", "entities"),
            ("item", "items"), ("element", "elements"), ("unit", "units"),
            ("type-A", "type-As"), ("type-B", "type-Bs"), ("type-C", "type-Cs"),
            ("category-X", "category-Xs"), ("category-Y", "category-Ys"),
            ("group-1", "group-1s"), ("group-2", "group-2s"),
        ]
        
        # Real-world entities (for diverse augmentation)
        self.real_entities = [
            # Animals
            ("dog", "dogs"), ("cat", "cats"), ("bird", "birds"), ("fish", "fish"),
            ("elephant", "elephants"), ("lion", "lions"), ("whale", "whales"),
            ("snake", "snakes"), ("frog", "frogs"), ("spider", "spiders"),
            ("eagle", "eagles"), ("penguin", "penguins"), ("bear", "bears"),
            # Objects
            ("book", "books"), ("chair", "chairs"), ("car", "cars"),
            ("phone", "phones"), ("computer", "computers"), ("table", "tables"),
            # People/Professions
            ("doctor", "doctors"), ("teacher", "teachers"), ("student", "students"),
            ("artist", "artists"), ("scientist", "scientists"), ("athlete", "athletes"),
            # Abstract categories
            ("mammal", "mammals"), ("reptile", "reptiles"), ("vehicle", "vehicles"),
            ("furniture", "furniture"), ("tool", "tools"), ("fruit", "fruits"),
        ]
    
    def _find_conclusion_marker(self, text: str) -> tuple:
        """Find conclusion marker position and the marker itself"""
        text_lower = text.lower()
        for marker in self.conclusion_markers:
            pos = text_lower.find(marker)
            if pos != -1:
                return pos, marker
        return None, None
    
    def _swap_premises(self, text: str) -> str:
        """Swap first two premises (preserves validity for most syllogisms)"""
        pos, marker = self._find_conclusion_marker(text)
        if pos is None:
            return text
        
        premises_part = text[:pos].strip()
        conclusion_part = text[pos:].strip()
        
        # Split by period
        sentences = [s.strip() for s in premises_part.split(".") if s.strip()]
        if len(sentences) < 2:
            return text
        
        # Swap first two
        sentences[0], sentences[1] = sentences[1], sentences[0]
        new_premises = ". ".join(sentences) + "."
        return f"{new_premises} {conclusion_part}"
    
    def _replace_quantifiers(self, text: str) -> str:
        """Replace quantifiers with synonyms"""
        words = text.split()
        result = []
        for w in words:
            w_lower = w.lower().strip(",.;:")
            if w_lower in self.quantifier_map and random.random() < self.quantifier_replace_prob:
                replacement = random.choice(self.quantifier_map[w_lower])
                # Preserve capitalization
                if w[0].isupper():
                    replacement = replacement.capitalize()
                result.append(w.replace(w_lower, replacement))
            else:
                result.append(w)
        return " ".join(result)
    
    def _replace_conclusion_marker(self, text: str) -> str:
        """Replace conclusion marker with synonym"""
        pos, old_marker = self._find_conclusion_marker(text)
        if pos is None:
            return text
        
        new_marker = random.choice(self.conclusion_markers)
        # Find the actual marker in original case
        actual_marker = text[pos:pos+len(old_marker)]
        if actual_marker[0].isupper():
            new_marker = new_marker.capitalize()
        
        return text[:pos] + new_marker + text[pos+len(old_marker):]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entity nouns from text"""
        # Simple heuristic: words after quantifiers
        entities = []
        words = text.lower().split()
        quantifiers = {"all", "every", "some", "no", "each", "any"}
        
        for i, w in enumerate(words):
            w_clean = w.strip(",.;:")
            if w_clean in quantifiers and i + 1 < len(words):
                next_word = words[i + 1].strip(",.;:")
                if next_word not in {"of", "the", "that", "which", "who"}:
                    entities.append(next_word)
        
        return list(set(entities))
    
    def _replace_entities(self, text: str, use_abstract: bool = False) -> str:
        """Replace entities with different ones (key debiasing technique)"""
        entities = self._extract_entities(text)
        if len(entities) < 2:
            return text
        
        # Choose replacement pool
        if use_abstract:
            pool = self.abstract_entities
        else:
            pool = self.real_entities
        
        # Sample replacements
        replacements = random.sample(pool, min(len(entities), len(pool)))
        
        result = text
        for i, entity in enumerate(entities[:len(replacements)]):
            singular, plural = replacements[i]
            # Replace both forms
            result = re.sub(rf'\b{entity}s\b', plural, result, flags=re.IGNORECASE)
            result = re.sub(rf'\b{entity}\b', singular, result, flags=re.IGNORECASE)
        
        return result
    
    def augment(self, text: str, use_abstract_entities: bool = False) -> str:
        """Apply random augmentations"""
        result = text
        
        if random.random() < self.swap_premises_prob:
            result = self._swap_premises(result)
        
        if random.random() < self.quantifier_replace_prob:
            result = self._replace_quantifiers(result)
        
        if random.random() < self.conclusion_marker_prob:
            result = self._replace_conclusion_marker(result)
        
        if random.random() < self.entity_replace_prob:
            result = self._replace_entities(result, use_abstract=use_abstract_entities)
        
        return result
    
    def augment_for_debiasing(self, text: str) -> str:
        """
        Stronger augmentation specifically for debiasing:
        Always replace entities to break content-validity correlation
        """
        result = text
        
        # Always do entity replacement with abstract entities
        result = self._replace_entities(result, use_abstract=True)
        
        # Also do other augmentations
        if random.random() < 0.5:
            result = self._swap_premises(result)
        if random.random() < 0.5:
            result = self._replace_quantifiers(result)
        
        return result


def apply_augmentation(
    samples: List[Dict], 
    augmenter: SyllogismAugmenter,
    multiplier: int = 3,
    hard_case_multiplier: int = 5,  # Extra augmentation for VI/IP
    use_debiasing_aug: bool = True,
) -> List[Dict]:
    """
    Apply augmentation with extra focus on hard cases (VI, IP).
    
    Args:
        samples: Original samples
        multiplier: Augmentation multiplier for easy cases
        hard_case_multiplier: Augmentation multiplier for VI/IP
        use_debiasing_aug: Use stronger entity replacement for debiasing
    """
    if multiplier <= 1 and hard_case_multiplier <= 1:
        return samples
    
    augmented = []
    
    for s in samples:
        v = s.get("validity")
        p = s.get("plausibility")
        
        # Determine quadrant
        if v is None or p is None:
            is_hard = False
        else:
            # Hard cases: Valid+Implausible or Invalid+Plausible
            is_hard = (v and not p) or (not v and p)
        
        # Determine multiplier
        actual_multiplier = hard_case_multiplier if is_hard else multiplier
        
        for k in range(actual_multiplier - 1):
            new_sample = dict(s)
            new_sample["id"] = f'{s.get("id", "")}_aug{k}'
            
            # Use debiasing augmentation for hard cases
            if use_debiasing_aug and is_hard:
                new_sample["syllogism"] = augmenter.augment_for_debiasing(s["syllogism"])
            else:
                new_sample["syllogism"] = augmenter.augment(s["syllogism"])
            
            augmented.append(new_sample)
    
    return samples + augmented


# =============================================================================
# Strategy 1: Template Abstraction
# =============================================================================
class TemplateExtractor:
    """
    Extract logical structure from syllogisms by replacing entities with symbols.
    E.g., "All dogs are mammals" -> "All A are B"
    """
    
    def __init__(self):
        # Quantifier patterns
        self.quantifiers = {
            "all": ["all", "every", "each", "any", "anything that is"],
            "some": ["some", "certain", "a few", "several", "a portion of", "a number of"],
            "no": ["no", "not any", "not a single", "none of the", "nothing that is"],
            "some_not": ["some .* are not", "not all", "at least one .* is not"]
        }
        
        # Conclusion markers
        self.conclusion_markers = [
            "therefore", "thus", "hence", "consequently", "so",
            "it follows that", "from this", "this proves", "this means",
            "the conclusion", "one must conclude", "it can be deduced",
            "it must be true", "it logically follows"
        ]
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract potential entity terms from syllogism"""
        text_lower = text.lower()
        
        # Remove conclusion markers
        for marker in self.conclusion_markers:
            idx = text_lower.find(marker)
            if idx != -1:
                text_lower = text_lower[:idx] + " [CONCLUSION] " + text_lower[idx + len(marker):]
        
        # Simple pattern: look for noun phrases after quantifiers
        entities = []
        
        # Pattern: "All X are Y" or "Some X are Y" or "No X is Y"
        patterns = [
            r"all\s+(\w+(?:\s+\w+)?)\s+are\s+(\w+(?:\s+\w+)?)",
            r"every\s+(\w+(?:\s+\w+)?)\s+is\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
            r"some\s+(\w+(?:\s+\w+)?)\s+are\s+(\w+(?:\s+\w+)?)",
            r"no\s+(\w+(?:\s+\w+)?)\s+is\s+(?:a\s+)?(\w+(?:\s+\w+)?)",
            r"there\s+(?:are|exist)\s+(?:some\s+)?(\w+(?:\s+\w+)?)\s+(?:that|which|who)\s+are\s+(\w+(?:\s+\w+)?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                entities.extend([m.strip() for m in match if m.strip()])
        
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            e_clean = e.strip().rstrip('s')  # Simple singular
            if e_clean not in seen and len(e_clean) > 1:
                seen.add(e_clean)
                unique_entities.append(e)
        
        return unique_entities[:3]  # At most 3 entities (A, B, C)
    
    def abstract(self, text: str) -> str:
        """Replace entities with abstract symbols"""
        entities = self.extract_entities(text)
        
        if len(entities) < 2:
            # Fallback: can't extract entities reliably
            return text
        
        symbols = ["A", "B", "C"]
        result = text
        
        for i, entity in enumerate(entities[:3]):
            if i < len(symbols):
                # Replace plural and singular forms
                result = re.sub(
                    rf'\b{re.escape(entity)}s?\b',
                    symbols[i],
                    result,
                    flags=re.IGNORECASE
                )
        
        return result


# =============================================================================
# Strategy 2: Gradient Reversal Layer (for Adversarial Debiasing)
# =============================================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# =============================================================================
# Strategy 5: Focal Loss for Hard Cases
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal loss with optional per-quadrant weighting.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be tensor of class weights
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets, quadrant_weights=None):
        ce_loss = F.cross_entropy(
            logits, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if quadrant_weights is not None:
            loss = loss * quadrant_weights
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        return loss.mean()


# =============================================================================
# Dataset with Plausibility-aware Features
# =============================================================================
class DebiasedSyllogismDataset(Dataset):
    """
    Dataset that supports:
    - Template abstraction (dual input)
    - Plausibility-aware SCL pairs
    - Quadrant information for weighted loss
    - Counterfactual pairs (same validity, different plausibility)
    """
    
    def __init__(
        self, 
        samples, 
        tokenizer, 
        max_length=256, 
        split="train",
        use_scl=False,
        use_template=False,
        template_extractor=None,
        plausibility_aware_scl=False,  # NEW: use plausibility for SCL pairing
        use_counterfactual=False,  # NEW: counterfactual consistency
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.use_scl = use_scl
        self.use_template = use_template
        self.template_extractor = template_extractor
        self.plausibility_aware_scl = plausibility_aware_scl
        self.use_counterfactual = use_counterfactual
        
        # Index samples by quadrant for SCL
        self.quadrant_indices = {
            "VP": [], "VI": [], "IP": [], "II": [], "unknown": []
        }
        for i, s in enumerate(samples):
            q = self._get_quadrant(s)
            self.quadrant_indices[q].append(i)
        
        # Also index by validity
        self.valid_indices = [i for i, s in enumerate(samples) if s.get("validity", True)]
        self.invalid_indices = [i for i, s in enumerate(samples) if not s.get("validity", True)]
    
    def _get_quadrant(self, sample) -> str:
        v = sample.get("validity", None)
        p = sample.get("plausibility", None)
        if v is None or p is None:
            return "unknown"
        elif v and p:
            return "VP"
        elif v and not p:
            return "VI"
        elif not v and p:
            return "IP"
        else:
            return "II"
    
    def _get_quadrant_weight(self, sample) -> float:
        """
        Higher weight for content-confusing cases:
        - Invalid + Plausible (IP): model might incorrectly predict valid
        - Valid + Implausible (VI): model might incorrectly predict invalid
        """
        q = self._get_quadrant(sample)
        weights = {
            "VP": 1.0,  # easy: both signals align
            "II": 1.0,  # easy: both signals align
            "VI": 2.0,  # hard: validity conflicts with plausibility
            "IP": 2.0,  # hard: validity conflicts with plausibility
            "unknown": 1.0,
        }
        return weights[q]
    
    def __len__(self):
        return len(self.samples)
    
    def _encode(self, text: str):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["syllogism"]
        label = 1 if s.get("validity", False) else 0
        plausibility = 1 if s.get("plausibility", True) else 0
        quadrant_weight = self._get_quadrant_weight(s)
        
        input_ids, attention_mask = self._encode(text)
        
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "plausibility": torch.tensor(plausibility, dtype=torch.long),
            "quadrant_weight": torch.tensor(quadrant_weight, dtype=torch.float),
            "id": s.get("id", ""),
            "source": s.get("_source", "unknown"),
        }
        
        # Add template version if enabled
        if self.use_template and self.template_extractor:
            template_text = self.template_extractor.abstract(text)
            template_ids, template_mask = self._encode(template_text)
            item["template_input_ids"] = template_ids
            item["template_attention_mask"] = template_mask
        
        # SCL pairs
        if self.split == "train" and self.use_scl:
            if self.plausibility_aware_scl:
                # Positive: same validity, different plausibility
                # This teaches the model that validity should be plausibility-invariant
                item.update(self._get_plausibility_aware_scl_pairs(idx, label))
            else:
                # Standard SCL: same validity = positive, different validity = negative
                item.update(self._get_standard_scl_pairs(idx, label))
        
        # Counterfactual pairs: same validity, different plausibility
        # Used to enforce consistency in logit space
        if self.split == "train" and self.use_counterfactual:
            cf_data = self._get_counterfactual_pair(idx, label)
            if cf_data is not None:
                item.update(cf_data)
        
        return item
    
    def _get_standard_scl_pairs(self, idx, label):
        """Standard SCL: positive = same validity, negative = different validity"""
        pos_pool = self.valid_indices if label == 1 else self.invalid_indices
        neg_pool = self.invalid_indices if label == 1 else self.valid_indices
        
        if len(pos_pool) > 1:
            pos_idx = random.choice([i for i in pos_pool if i != idx])
        else:
            pos_idx = idx
        neg_idx = random.choice(neg_pool) if neg_pool else idx
        
        pos_ids, pos_mask = self._encode(self.samples[pos_idx]["syllogism"])
        neg_ids, neg_mask = self._encode(self.samples[neg_idx]["syllogism"])
        
        return {
            "pos_input_ids": pos_ids,
            "pos_attention_mask": pos_mask,
            "neg_input_ids": neg_ids,
            "neg_attention_mask": neg_mask,
        }
    
    def _get_plausibility_aware_scl_pairs(self, idx, label):
        """
        Plausibility-aware SCL:
        - Positive: same validity, DIFFERENT plausibility
          (teaches model: VI and VP should have similar representations)
        - Negative: different validity
        """
        current_quadrant = self._get_quadrant(self.samples[idx])
        
        # Find positive: same validity, different plausibility
        if current_quadrant in ["VP", "VI"]:  # Valid
            # Positive is valid with opposite plausibility
            opposite = "VI" if current_quadrant == "VP" else "VP"
            pos_pool = self.quadrant_indices[opposite]
            neg_quadrants = ["IP", "II"]  # Invalid
        else:  # Invalid
            opposite = "II" if current_quadrant == "IP" else "IP"
            pos_pool = self.quadrant_indices[opposite]
            neg_quadrants = ["VP", "VI"]  # Valid
        
        # Fallback if opposite plausibility quadrant is empty
        if not pos_pool:
            pos_pool = self.valid_indices if label == 1 else self.invalid_indices
        
        neg_pool = []
        for q in neg_quadrants:
            neg_pool.extend(self.quadrant_indices[q])
        if not neg_pool:
            neg_pool = self.invalid_indices if label == 1 else self.valid_indices
        
        pos_idx = random.choice(pos_pool) if pos_pool else idx
        neg_idx = random.choice(neg_pool) if neg_pool else idx
        
        pos_ids, pos_mask = self._encode(self.samples[pos_idx]["syllogism"])
        neg_ids, neg_mask = self._encode(self.samples[neg_idx]["syllogism"])
        
        return {
            "pos_input_ids": pos_ids,
            "pos_attention_mask": pos_mask,
            "neg_input_ids": neg_ids,
            "neg_attention_mask": neg_mask,
        }
    
    def _get_counterfactual_pair(self, idx, label):
        """
        Get a counterfactual pair: same validity, DIFFERENT plausibility.
        This is used to enforce that the model produces similar logits
        regardless of content plausibility.
        
        Returns:
            dict with cf_input_ids, cf_attention_mask, cf_label
            or None if no valid counterfactual exists
        """
        current_quadrant = self._get_quadrant(self.samples[idx])
        
        if current_quadrant == "unknown":
            return None
        
        # Find counterfactual: same validity, opposite plausibility
        if current_quadrant == "VP":
            cf_pool = self.quadrant_indices["VI"]
        elif current_quadrant == "VI":
            cf_pool = self.quadrant_indices["VP"]
        elif current_quadrant == "IP":
            cf_pool = self.quadrant_indices["II"]
        elif current_quadrant == "II":
            cf_pool = self.quadrant_indices["IP"]
        else:
            return None
        
        if not cf_pool:
            return None
        
        cf_idx = random.choice(cf_pool)
        cf_sample = self.samples[cf_idx]
        cf_ids, cf_mask = self._encode(cf_sample["syllogism"])
        cf_label = 1 if cf_sample.get("validity", False) else 0
        
        return {
            "cf_input_ids": cf_ids,
            "cf_attention_mask": cf_mask,
            "cf_label": torch.tensor(cf_label, dtype=torch.long),
        }


# =============================================================================
# Model with Adversarial Debiasing
# =============================================================================
class DebiasedDeBERTaClassifier(nn.Module):
    """
    DeBERTa classifier with:
    - Optional template fusion
    - Adversarial plausibility head with GRL
    - SCL projection head
    - Focal loss support
    - Counterfactual consistency loss
    """
    
    def __init__(
        self,
        model_name="microsoft/deberta-v3-large",
        num_labels=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.15,
        lora_targets=("query_proj", "key_proj", "value_proj", "dense"),
        # Loss settings
        use_focal_loss=False,
        focal_gamma=2.0,
        use_label_smoothing=True,
        label_smoothing=0.1,
        # SCL settings
        use_scl=True,
        scl_temp=0.07,
        scl_proj_dim=256,
        scl_weight=0.5,
        # Adversarial debiasing settings
        use_adversarial=False,
        adv_weight=0.3,
        grl_alpha=1.0,
        # Template fusion
        use_template_fusion=False,
        template_fusion_type="concat",  # "concat", "add", or "attention"
        # Counterfactual consistency
        use_counterfactual=False,
        cf_weight=0.5,
    ):
        super().__init__()
        self.use_scl = use_scl
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight
        self.use_adversarial = use_adversarial
        self.adv_weight = adv_weight
        self.use_template_fusion = use_template_fusion
        self.template_fusion_type = template_fusion_type
        self.use_focal_loss = use_focal_loss
        self.use_counterfactual = use_counterfactual
        self.cf_weight = cf_weight
        
        # Base model
        self.deberta = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        # LoRA
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias="none",
        )
        self.deberta = get_peft_model(self.deberta, peft_cfg)
        
        h = self.deberta.config.hidden_size
        
        # Template fusion layer
        if use_template_fusion:
            if template_fusion_type == "concat":
                self.fusion = nn.Sequential(
                    nn.Linear(h * 2, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
            elif template_fusion_type == "attention":
                self.fusion_attn = nn.MultiheadAttention(h, num_heads=8, dropout=0.1, batch_first=True)
                self.fusion_norm = nn.LayerNorm(h)
        
        # Classifier head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h, num_labels)
        
        # Loss
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        elif use_label_smoothing:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # SCL projection
        if use_scl:
            self.projection = nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h, scl_proj_dim),
            )
        
        # Adversarial plausibility head (with GRL)
        if use_adversarial:
            self.grl = GradientReversalLayer(alpha=grl_alpha)
            self.plausibility_classifier = nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h // 2, 2),  # plausible / implausible
            )
    
    def encode_cls(self, input_ids, attention_mask):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        return pooled
    
    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        plausibility=None,
        quadrant_weight=None,
        template_input_ids=None,
        template_attention_mask=None,
        use_rdrop=False,
        rdrop_alpha=0.7,
        pos_input_ids=None,
        pos_attention_mask=None,
        neg_input_ids=None,
        neg_attention_mask=None,
        # Counterfactual inputs
        cf_input_ids=None,
        cf_attention_mask=None,
        cf_label=None,
    ):
        # Encode main input
        pooled = self.encode_cls(input_ids, attention_mask)
        
        # Template fusion
        if self.use_template_fusion and template_input_ids is not None:
            template_pooled = self.encode_cls(template_input_ids, template_attention_mask)
            
            if self.template_fusion_type == "concat":
                fused = torch.cat([pooled, template_pooled], dim=-1)
                pooled = self.fusion(fused)
            elif self.template_fusion_type == "add":
                pooled = pooled + template_pooled
            elif self.template_fusion_type == "attention":
                # Cross-attention: query from main, key/value from template
                pooled_exp = pooled.unsqueeze(1)  # [B, 1, H]
                template_exp = template_pooled.unsqueeze(1)
                attn_out, _ = self.fusion_attn(pooled_exp, template_exp, template_exp)
                pooled = self.fusion_norm(pooled + attn_out.squeeze(1))
        
        logits = self.classifier(pooled)
        
        loss = None
        loss_dict = {}
        
        if labels is not None:
            # Main classification loss
            if self.use_focal_loss and quadrant_weight is not None:
                ce = self.criterion(logits, labels, quadrant_weight)
            else:
                ce = self.criterion(logits, labels)
            loss = ce
            loss_dict["ce_loss"] = ce.item()
            
            # R-Drop
            if use_rdrop and self.training:
                pooled2 = self.encode_cls(input_ids, attention_mask)
                if self.use_template_fusion and template_input_ids is not None:
                    template_pooled2 = self.encode_cls(template_input_ids, template_attention_mask)
                    if self.template_fusion_type == "concat":
                        fused2 = torch.cat([pooled2, template_pooled2], dim=-1)
                        pooled2 = self.fusion(fused2)
                    elif self.template_fusion_type == "add":
                        pooled2 = pooled2 + template_pooled2
                
                logits2 = self.classifier(pooled2)
                ce2 = self.criterion(logits2, labels) if not self.use_focal_loss else self.criterion(logits2, labels, quadrant_weight)
                
                kl1 = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction="batchmean")
                kl2 = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean")
                kl = 0.5 * (kl1 + kl2)
                loss = 0.5 * (ce + ce2) + rdrop_alpha * kl
                loss_dict["rdrop_kl"] = kl.item()
            
            # SCL loss
            if self.use_scl and self.training and pos_input_ids is not None and neg_input_ids is not None:
                pos_pooled = self.encode_cls(pos_input_ids, pos_attention_mask)
                neg_pooled = self.encode_cls(neg_input_ids, neg_attention_mask)
                
                z = F.normalize(self.projection(pooled), dim=-1)
                z_pos = F.normalize(self.projection(pos_pooled), dim=-1)
                z_neg = F.normalize(self.projection(neg_pooled), dim=-1)
                
                sim_pos = (z * z_pos).sum(dim=-1) / self.scl_temp
                sim_neg = (z * z_neg).sum(dim=-1) / self.scl_temp
                scl_logits = torch.stack([sim_pos, sim_neg], dim=1)
                scl_labels = torch.zeros(scl_logits.size(0), dtype=torch.long, device=scl_logits.device)
                scl_loss = F.cross_entropy(scl_logits, scl_labels)
                
                loss = loss + self.scl_weight * scl_loss
                loss_dict["scl_loss"] = scl_loss.item()
            
            # Adversarial plausibility loss
            if self.use_adversarial and self.training and plausibility is not None:
                # Apply GRL - gradients will be reversed
                reversed_pooled = self.grl(pooled)
                plaus_logits = self.plausibility_classifier(reversed_pooled)
                adv_loss = F.cross_entropy(plaus_logits, plausibility)
                
                loss = loss + self.adv_weight * adv_loss
                loss_dict["adv_loss"] = adv_loss.item()
            
            # Counterfactual consistency loss
            # Enforce that samples with same validity but different plausibility
            # should produce similar logits (plausibility shouldn't affect validity prediction)
            if self.use_counterfactual and self.training and cf_input_ids is not None:
                # Get logits for counterfactual sample
                cf_pooled = self.encode_cls(cf_input_ids, cf_attention_mask)
                cf_logits = self.classifier(cf_pooled)
                
                # The key insight: both should predict the SAME validity
                # We use MSE on the logits to enforce consistency
                # Only compute for samples where cf exists (cf_label should match labels)
                
                # Soft consistency: MSE between logit distributions
                cf_loss = F.mse_loss(logits, cf_logits)
                
                # Alternative: KL divergence for softer constraint
                # cf_loss = 0.5 * (
                #     F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(cf_logits, dim=-1), reduction='batchmean') +
                #     F.kl_div(F.log_softmax(cf_logits, dim=-1), F.softmax(logits, dim=-1), reduction='batchmean')
                # )
                
                loss = loss + self.cf_weight * cf_loss
                loss_dict["cf_loss"] = cf_loss.item()
        
        return {"loss": loss, "logits": logits, "loss_dict": loss_dict}


# =============================================================================
# Strategy 3: Quadrant-balanced Sampling
# =============================================================================
def build_quadrant_balanced_sampler(samples, hard_case_boost=2.0):
    """
    Create sampler that balances across all 4 quadrants,
    with extra weight on hard cases (VI, IP).
    """
    quadrant_counts = {"VP": 0, "VI": 0, "IP": 0, "II": 0, "unknown": 0}
    sample_quadrants = []
    
    for s in samples:
        v = s.get("validity", None)
        p = s.get("plausibility", None)
        if v is None or p is None:
            q = "unknown"
        elif v and p:
            q = "VP"
        elif v and not p:
            q = "VI"
        elif not v and p:
            q = "IP"
        else:
            q = "II"
        quadrant_counts[q] += 1
        sample_quadrants.append(q)
    
    # Compute weights: inverse frequency * hard case boost
    total = len(samples)
    weights = []
    for q in sample_quadrants:
        if quadrant_counts[q] > 0:
            base_weight = total / (4 * quadrant_counts[q])  # Inverse frequency
        else:
            base_weight = 1.0
        
        # Boost hard cases
        if q in ["VI", "IP"]:
            base_weight *= hard_case_boost
        
        weights.append(base_weight)
    
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# =============================================================================
# Training / Evaluation
# =============================================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1, use_rdrop=False, rdrop_alpha=0.7, use_adversarial=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc="Training")
    step = 0
    
    for batch in pbar:
        step += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Only pass plausibility if adversarial is enabled this epoch
        plausibility = None
        if use_adversarial:
            plausibility = batch.get("plausibility")
            if plausibility is not None:
                plausibility = plausibility.to(device)
        
        quadrant_weight = batch.get("quadrant_weight")
        if quadrant_weight is not None:
            quadrant_weight = quadrant_weight.to(device)
        
        kwargs = {
            "plausibility": plausibility,
            "quadrant_weight": quadrant_weight,
        }
        
        # Template inputs
        if "template_input_ids" in batch:
            kwargs["template_input_ids"] = batch["template_input_ids"].to(device)
            kwargs["template_attention_mask"] = batch["template_attention_mask"].to(device)
        
        # SCL pairs
        if "pos_input_ids" in batch:
            kwargs.update({
                "pos_input_ids": batch["pos_input_ids"].to(device),
                "pos_attention_mask": batch["pos_attention_mask"].to(device),
                "neg_input_ids": batch["neg_input_ids"].to(device),
                "neg_attention_mask": batch["neg_attention_mask"].to(device),
            })
        
        # Counterfactual pairs
        if "cf_input_ids" in batch:
            kwargs.update({
                "cf_input_ids": batch["cf_input_ids"].to(device),
                "cf_attention_mask": batch["cf_attention_mask"].to(device),
                "cf_label": batch["cf_label"].to(device),
            })
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_rdrop=use_rdrop,
            rdrop_alpha=rdrop_alpha,
            **kwargs
        )
        
        loss = outputs["loss"] / grad_accum
        loss.backward()
        
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * grad_accum
        
        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({"loss": f"{(loss.item()*grad_accum):.4f}"})
    
    if step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    # Per-quadrant metrics
    quadrant_stats = {
        "VP": {"correct": 0, "total": 0},
        "VI": {"correct": 0, "total": 0},
        "IP": {"correct": 0, "total": 0},
        "II": {"correct": 0, "total": 0},
    }
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        plausibility = batch.get("plausibility", torch.ones_like(labels)).to(device)
        
        kwargs = {}
        if "template_input_ids" in batch:
            kwargs["template_input_ids"] = batch["template_input_ids"].to(device)
            kwargs["template_attention_mask"] = batch["template_attention_mask"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        total_loss += outputs["loss"].item()
        
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Track per-quadrant accuracy
        for i in range(labels.size(0)):
            v = labels[i].item()
            p = plausibility[i].item()
            pred_correct = (preds[i] == labels[i]).item()
            
            if v == 1 and p == 1:
                q = "VP"
            elif v == 1 and p == 0:
                q = "VI"
            elif v == 0 and p == 1:
                q = "IP"
            else:
                q = "II"
            
            quadrant_stats[q]["total"] += 1
            quadrant_stats[q]["correct"] += int(pred_correct)
    
    # Compute quadrant accuracies
    quadrant_accs = {}
    for q, stats in quadrant_stats.items():
        if stats["total"] > 0:
            quadrant_accs[q] = stats["correct"] / stats["total"]
        else:
            quadrant_accs[q] = 0.0
    
    # Detect degenerate predictions (model predicts all same class)
    valid_acc = (quadrant_accs["VP"] + quadrant_accs["VI"]) / 2 if (quadrant_stats["VP"]["total"] > 0 and quadrant_stats["VI"]["total"] > 0) else 0
    invalid_acc = (quadrant_accs["IP"] + quadrant_accs["II"]) / 2 if (quadrant_stats["IP"]["total"] > 0 and quadrant_stats["II"]["total"] > 0) else 0
    
    is_degenerate = (valid_acc < 0.1 or invalid_acc < 0.1)  # Model predicts almost all one class
    
    # Compute Content Effect (TCE) approximation
    # TCE = average |acc_plausible - acc_implausible| across validity conditions
    valid_diff = abs(quadrant_accs["VP"] - quadrant_accs["VI"]) if (quadrant_stats["VP"]["total"] > 0 and quadrant_stats["VI"]["total"] > 0) else 0
    invalid_diff = abs(quadrant_accs["IP"] - quadrant_accs["II"]) if (quadrant_stats["IP"]["total"] > 0 and quadrant_stats["II"]["total"] > 0) else 0
    tce_approx = (valid_diff + invalid_diff) / 2 * 100  # As percentage
    
    # Combined score approximation
    acc = correct / max(total, 1) * 100
    
    # If degenerate, penalize heavily (set combined to 0)
    if is_degenerate:
        combined_score = 0.0
        tce_approx = 100.0  # Mark as bad
    else:
        combined_score = acc / (1 + math.log(1 + tce_approx)) if tce_approx >= 0 else acc
    
    # Also compute min quadrant accuracy (useful metric)
    min_quadrant_acc = min(quadrant_accs["VP"], quadrant_accs["VI"], quadrant_accs["IP"], quadrant_accs["II"])
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": correct / max(total, 1),
        "acc_VP": quadrant_accs["VP"],
        "acc_VI": quadrant_accs["VI"],
        "acc_IP": quadrant_accs["IP"],
        "acc_II": quadrant_accs["II"],
        "valid_class_acc": valid_acc,
        "invalid_class_acc": invalid_acc,
        "min_quadrant_acc": min_quadrant_acc,
        "is_degenerate": is_degenerate,
        "tce_approx": tce_approx,
        "combined_score_approx": combined_score,
    }


# =============================================================================
# Data Loading
# =============================================================================
def load_data(path, source_tag):
    data = load_json_list(path)
    samples = []
    for item in data:
        samples.append({
            "id": item.get("id", ""),
            "syllogism": item["syllogism"],
            "validity": bool(item.get("validity", item.get("label", False))),
            "plausibility": item.get("plausibility", None),
            "_source": source_tag,
        })
    return samples


# =============================================================================
# Main Training Function
# =============================================================================
def run_once(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("SemEval 2026 Task 11 - Subtask 1 (Content-Debiased)")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"\n[Debiasing Strategies]")
    print(f"  Focal Loss: {args.use_focal_loss} (gamma={args.focal_gamma})")
    print(f"  Adversarial Debiasing: {args.use_adversarial} (weight={args.adv_weight}, grl_alpha={args.grl_alpha}, warmup={args.adv_warmup_epochs} epochs)")
    print(f"  Quadrant-balanced Sampling: {args.use_quadrant_sampling} (hard_boost={args.hard_case_boost})")
    print(f"  Plausibility-aware SCL: {args.plausibility_aware_scl}")
    print(f"  Template Abstraction: {args.use_template}")
    print(f"  Counterfactual Consistency: {args.use_counterfactual} (weight={args.cf_weight})")
    print(f"  Data Augmentation: {args.use_augmentation} (easy x{args.aug_multiplier}, hard x{args.hard_case_aug_multiplier}, debiasing={args.use_debiasing_aug})")
    print(f"\n[Standard Settings]")
    print(f"  SCL: {args.use_scl} (w={args.scl_weight}, temp={args.scl_temp})")
    print(f"  R-Drop: {args.use_rdrop} (alpha={args.rdrop_alpha})")
    print(f"  Label smoothing: {args.use_label_smoothing} (eps={args.label_smoothing})")
    print(f"  Seed: {args.seed}")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    official_samples = load_data(args.official_path, source_tag="official")
    print_quadrant_distribution(official_samples, "Official data")
    
    # Split
    if args.mode == "dev":
        train_ratio = 1.0 - args.val_split
        train_samples, val_samples = stratified_split(official_samples, train_ratio=train_ratio, seed=args.seed)
        print_quadrant_distribution(train_samples, "Train set (before aug)")
        print_quadrant_distribution(val_samples, "Val set")
    else:
        train_samples = official_samples
        val_samples = None
    
    # Data augmentation
    if args.use_augmentation:
        augmenter = SyllogismAugmenter(
            swap_premises_prob=args.swap_premises_prob,
            quantifier_replace_prob=args.quantifier_replace_prob,
            entity_replace_prob=args.entity_replace_prob,
            seed=args.seed,
        )
        before_count = len(train_samples)
        train_samples = apply_augmentation(
            train_samples,
            augmenter,
            multiplier=args.aug_multiplier,
            hard_case_multiplier=args.hard_case_aug_multiplier,
            use_debiasing_aug=args.use_debiasing_aug,
        )
        print(f"\n[Augmentation] {before_count} -> {len(train_samples)} samples")
        print(f"  Easy cases: x{args.aug_multiplier}, Hard cases (VI/IP): x{args.hard_case_aug_multiplier}")
        if args.use_debiasing_aug:
            print(f"  Using abstract entity replacement for hard cases")
        print_quadrant_distribution(train_samples, "Train set (after aug)")
    
    # Tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Template extractor
    template_extractor = TemplateExtractor() if args.use_template else None
    
    # Datasets
    print("\n[3/5] Creating datasets...")
    train_dataset = DebiasedSyllogismDataset(
        train_samples, 
        tokenizer, 
        max_length=args.max_length, 
        split="train",
        use_scl=args.use_scl,
        use_template=args.use_template,
        template_extractor=template_extractor,
        plausibility_aware_scl=args.plausibility_aware_scl,
        use_counterfactual=args.use_counterfactual,
    )
    
    # Sampler
    if args.use_quadrant_sampling:
        sampler = build_quadrant_balanced_sampler(train_samples, hard_case_boost=args.hard_case_boost)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_loader = None
    if args.mode == "dev":
        val_dataset = DebiasedSyllogismDataset(
            val_samples, 
            tokenizer, 
            max_length=args.max_length, 
            split="val",
            use_scl=False,
            use_template=args.use_template,
            template_extractor=template_extractor,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    print("\n[4/5] Creating model...")
    lora_targets = tuple([x.strip() for x in args.lora_targets.split(",") if x.strip()])
    model = DebiasedDeBERTaClassifier(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=lora_targets,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_label_smoothing=args.use_label_smoothing,
        label_smoothing=args.label_smoothing,
        use_scl=args.use_scl,
        scl_temp=args.scl_temp,
        scl_proj_dim=args.scl_proj_dim,
        scl_weight=args.scl_weight,
        use_adversarial=args.use_adversarial,
        adv_weight=args.adv_weight,
        grl_alpha=args.grl_alpha,
        use_template_fusion=args.use_template and args.template_fusion,
        template_fusion_type=args.template_fusion_type,
        use_counterfactual=args.use_counterfactual,
        cf_weight=args.cf_weight,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Training
    print("\n[5/5] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    best_combined = 0.0
    best_acc = 0.0
    no_improve = 0
    history = []
    
    for epoch in range(args.epochs):
        # Adversarial warmup: don't use adversarial loss in early epochs
        use_adv_this_epoch = args.use_adversarial and (epoch >= args.adv_warmup_epochs)
        if args.use_adversarial and epoch == args.adv_warmup_epochs:
            print(f"\n  ⚡ Enabling adversarial debiasing from this epoch")
        
        print(f"\nEpoch {epoch+1}/{args.epochs}" + (" [adv warmup]" if args.use_adversarial and not use_adv_this_epoch else ""))
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum=args.grad_accum,
            use_rdrop=args.use_rdrop,
            rdrop_alpha=args.rdrop_alpha,
            use_adversarial=use_adv_this_epoch,  # Pass flag to training
        )
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        if args.mode == "dev" and val_loader:
            metrics = evaluate(model, val_loader, device)
            print(f"  Val Loss: {metrics['loss']:.4f} | Val Acc: {metrics['accuracy']:.4f}")
            print(f"  Quadrant Acc: VP={metrics['acc_VP']:.3f}, VI={metrics['acc_VI']:.3f}, IP={metrics['acc_IP']:.3f}, II={metrics['acc_II']:.3f}")
            print(f"  Min Quadrant Acc: {metrics['min_quadrant_acc']:.3f} | Valid/Invalid class acc: {metrics['valid_class_acc']:.3f}/{metrics['invalid_class_acc']:.3f}")
            
            if metrics["is_degenerate"]:
                print(f"  ⚠️  DEGENERATE: Model predicting mostly one class!")
                print(f"  TCE (approx): N/A | Combined (approx): 0.00 (penalized)")
            else:
                print(f"  TCE (approx): {metrics['tce_approx']:.2f} | Combined (approx): {metrics['combined_score_approx']:.2f}")
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": metrics["loss"],
                "val_acc": metrics["accuracy"],
                "acc_VP": metrics["acc_VP"],
                "acc_VI": metrics["acc_VI"],
                "acc_IP": metrics["acc_IP"],
                "acc_II": metrics["acc_II"],
                "min_quadrant_acc": metrics["min_quadrant_acc"],
                "is_degenerate": metrics["is_degenerate"],
                "tce_approx": metrics["tce_approx"],
                "combined_score_approx": metrics["combined_score_approx"],
            })
            
            # Use combined score for early stopping, but only for non-degenerate models
            # Also require minimum accuracy threshold
            current_score = metrics["combined_score_approx"]
            is_valid_model = not metrics["is_degenerate"] and metrics["accuracy"] > 0.6
            
            if is_valid_model and current_score > best_combined:
                best_combined = current_score
                best_acc = metrics["accuracy"]
                torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
                print(f"  ✓ Saved best model (combined: {best_combined:.2f}, acc: {best_acc:.4f})")
                no_improve = 0
            else:
                no_improve += 1
                if metrics["is_degenerate"]:
                    print(f"  Waiting for model to converge... ({no_improve}/{args.patience})")
                else:
                    print(f"  Early stopping: {no_improve}/{args.patience}")
                if no_improve >= args.patience:
                    print("\nEarly stopping triggered.")
                    break
        else:
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})
            torch.save(model.state_dict(), f"{args.output_dir}/model_epoch{epoch+1}.pt")
    
    if args.mode == "final":
        torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
    
    with open(f"{args.output_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    config = vars(args)
    config["best_combined_score"] = best_combined
    config["best_accuracy"] = best_acc
    config["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best combined score (approx): {best_combined:.2f}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {args.output_dir}/best_model.pt")
    print("=" * 70)
    
    return best_combined


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "final"])
    parser.add_argument("--official_path", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    
    # Training basics
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=7)
    
    # Model / LoRA
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.15)
    parser.add_argument("--lora_targets", type=str, default="query_proj,key_proj,value_proj,dense")
    
    # Standard regularization
    parser.add_argument("--use_label_smoothing", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_rdrop", action="store_true")
    parser.add_argument("--rdrop_alpha", type=float, default=0.7)
    
    # SCL
    parser.add_argument("--use_scl", action="store_true")
    parser.add_argument("--scl_temp", type=float, default=0.07)
    parser.add_argument("--scl_proj_dim", type=int, default=256)
    parser.add_argument("--scl_weight", type=float, default=0.5)
    
    # === NEW: Debiasing strategies ===
    # Strategy 1: Template abstraction
    parser.add_argument("--use_template", action="store_true", help="Use template abstraction")
    parser.add_argument("--template_fusion", action="store_true", help="Fuse template with original")
    parser.add_argument("--template_fusion_type", type=str, default="concat", choices=["concat", "add", "attention"])
    
    # Strategy 2: Adversarial debiasing
    parser.add_argument("--use_adversarial", action="store_true", help="Use adversarial plausibility head")
    parser.add_argument("--adv_weight", type=float, default=0.2, help="Weight for adversarial loss")
    parser.add_argument("--grl_alpha", type=float, default=0.5, help="Gradient reversal strength")
    parser.add_argument("--adv_warmup_epochs", type=int, default=2, help="Epochs before enabling adversarial loss")
    
    # Strategy 3: Quadrant-balanced sampling
    parser.add_argument("--use_quadrant_sampling", action="store_true", help="Balance across 4 quadrants")
    parser.add_argument("--hard_case_boost", type=float, default=2.0, help="Boost factor for VI/IP")
    
    # Strategy 4: Plausibility-aware SCL
    parser.add_argument("--plausibility_aware_scl", action="store_true", help="SCL: same validity, different plausibility = positive")
    
    # Strategy 5: Focal loss
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    # Strategy 6: Counterfactual consistency (NEW)
    parser.add_argument("--use_counterfactual", action="store_true", help="Use counterfactual consistency loss")
    parser.add_argument("--cf_weight", type=float, default=0.5, help="Weight for counterfactual consistency loss")
    
    # Data augmentation (critical for debiasing)
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--aug_multiplier", type=int, default=3, help="Augmentation multiplier for easy cases")
    parser.add_argument("--hard_case_aug_multiplier", type=int, default=5, help="Augmentation multiplier for VI/IP")
    parser.add_argument("--use_debiasing_aug", action="store_true", help="Use abstract entity replacement for hard cases")
    parser.add_argument("--swap_premises_prob", type=float, default=0.3)
    parser.add_argument("--quantifier_replace_prob", type=float, default=0.3)
    parser.add_argument("--entity_replace_prob", type=float, default=0.5)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="models/subtask1_debiased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    
    args = parser.parse_args()
    
    seeds = parse_seeds(args.seeds)
    if seeds:
        base_out = args.output_dir
        os.makedirs(base_out, exist_ok=True)
        
        print("=" * 70)
        print(f"[Multi-seed] seeds={seeds}")
        print("=" * 70)
        
        results = []
        for sd in seeds:
            run_args = copy.deepcopy(args)
            run_args.seed = sd
            run_args.output_dir = os.path.join(base_out, f"seed{sd}")
            best = run_once(run_args)
            results.append(best)
        
        mean_score = statistics.mean(results)
        std_score = statistics.pstdev(results) if len(results) > 1 else 0.0
        
        summary = {
            "seeds": seeds,
            "best_combined": results,
            "mean": mean_score,
            "std": std_score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(base_out, "multi_seed_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 70)
        print("[Multi-seed] Done.")
        for sd, score in zip(seeds, results):
            print(f"  seed={sd}: combined={score:.2f}")
        print(f"  mean={mean_score:.2f}  std={std_score:.2f}")
        print("=" * 70)
        return
    
    run_once(args)


if __name__ == "__main__":
    main()
