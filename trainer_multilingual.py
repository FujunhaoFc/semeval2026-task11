#!/usr/bin/env python3
"""
SemEval 2026 Task 11 - Subtask 3 Multilingual Trainer

Based on trainer_debiased.py, adapted for:
1. XLM-RoBERTa (for low-resource language support)
2. Multilingual zero-shot transfer
3. Compatible with DeBERTa/mDeBERTa as fallback

Key changes from original:
- Auto-detect LoRA target modules based on model architecture
- Support for xlm-roberta-large, mdeberta-v3-base, deberta-v3-large
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
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


# =============================================================================
# Model Architecture Detection
# =============================================================================
def get_lora_target_modules(model_name: str) -> List[str]:
    """
    Auto-detect LoRA target modules based on model architecture.
    
    Different models use different naming conventions:
    - DeBERTa: query_proj, key_proj, value_proj, dense
    - RoBERTa/XLM-R: query, key, value, dense
    - mDeBERTa: Same as DeBERTa
    """
    model_name_lower = model_name.lower()
    
    if "deberta" in model_name_lower:
        # DeBERTa and mDeBERTa
        return ["query_proj", "key_proj", "value_proj", "dense"]
    elif "roberta" in model_name_lower or "xlm" in model_name_lower:
        # RoBERTa, XLM-RoBERTa
        return ["query", "key", "value", "dense"]
    elif "bert" in model_name_lower:
        # BERT variants
        return ["query", "key", "value", "dense"]
    else:
        # Default fallback
        print(f"Warning: Unknown model architecture for {model_name}, using default LoRA targets")
        return ["query", "key", "value", "dense"]


def get_model_info(model_name: str) -> dict:
    """Get model-specific information"""
    info = {
        "xlm-roberta-large": {
            "hidden_size": 1024,
            "languages": "100+",
            "low_resource": True,  # Good for Bengali, Telugu, Swahili
        },
        "xlm-roberta-base": {
            "hidden_size": 768,
            "languages": "100+",
            "low_resource": True,
        },
        "microsoft/mdeberta-v3-base": {
            "hidden_size": 768,
            "languages": "~50",
            "low_resource": False,  # Limited for Bengali, Telugu, Swahili
        },
        "microsoft/deberta-v3-large": {
            "hidden_size": 1024,
            "languages": "English only",
            "low_resource": False,
        },
    }
    
    for key in info:
        if key in model_name.lower():
            return info[key]
    
    return {"hidden_size": "auto", "languages": "unknown", "low_resource": "unknown"}


# =============================================================================
# Utils (same as original)
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
# Data Augmentation (same as original)
# =============================================================================
class SyllogismAugmenter:
    def __init__(
        self, 
        swap_premises_prob=0.3, 
        quantifier_replace_prob=0.3,
        entity_replace_prob=0.5,
        conclusion_marker_prob=0.3,
        seed=42
    ):
        random.seed(seed)
        self.swap_premises_prob = swap_premises_prob
        self.quantifier_replace_prob = quantifier_replace_prob
        self.entity_replace_prob = entity_replace_prob
        self.conclusion_marker_prob = conclusion_marker_prob
        
        self.quantifier_map = {
            "all": ["every", "each", "any"],
            "every": ["all", "each", "any"],
            "some": ["certain", "a few", "several"],
            "no": ["not any", "not a single", "none of the"],
        }
        
        self.conclusion_markers = [
            "therefore", "thus", "hence", "consequently", "so",
            "it follows that", "from this", "this means that",
            "we can conclude that", "it must be that"
        ]
        
        self.abstract_entities = [
            ("alpha", "alphas"), ("beta", "betas"), ("gamma", "gammas"),
            ("zeta", "zetas"), ("omega", "omegas"), ("delta", "deltas"),
            ("thing", "things"), ("object", "objects"), ("entity", "entities"),
            ("item", "items"), ("element", "elements"), ("unit", "units"),
        ]
        
        self.real_entities = [
            ("dog", "dogs"), ("cat", "cats"), ("bird", "birds"),
            ("book", "books"), ("chair", "chairs"), ("car", "cars"),
            ("doctor", "doctors"), ("teacher", "teachers"), ("student", "students"),
            ("mammal", "mammals"), ("reptile", "reptiles"), ("vehicle", "vehicles"),
        ]
    
    def _find_conclusion_marker(self, text: str) -> tuple:
        text_lower = text.lower()
        for marker in self.conclusion_markers:
            pos = text_lower.find(marker)
            if pos != -1:
                return pos, marker
        return None, None
    
    def _swap_premises(self, text: str) -> str:
        pos, marker = self._find_conclusion_marker(text)
        if pos is None:
            return text
        
        premises_part = text[:pos].strip()
        conclusion_part = text[pos:].strip()
        
        sentences = [s.strip() for s in premises_part.split(".") if s.strip()]
        if len(sentences) < 2:
            return text
        
        sentences[0], sentences[1] = sentences[1], sentences[0]
        new_premises = ". ".join(sentences) + "."
        return f"{new_premises} {conclusion_part}"
    
    def _replace_quantifiers(self, text: str) -> str:
        words = text.split()
        result = []
        for w in words:
            w_lower = w.lower().strip(",.;:")
            if w_lower in self.quantifier_map and random.random() < self.quantifier_replace_prob:
                replacement = random.choice(self.quantifier_map[w_lower])
                if w[0].isupper():
                    replacement = replacement.capitalize()
                result.append(w.replace(w_lower, replacement))
            else:
                result.append(w)
        return " ".join(result)
    
    def _replace_conclusion_marker(self, text: str) -> str:
        pos, old_marker = self._find_conclusion_marker(text)
        if pos is None:
            return text
        
        new_marker = random.choice(self.conclusion_markers)
        actual_marker = text[pos:pos+len(old_marker)]
        if actual_marker[0].isupper():
            new_marker = new_marker.capitalize()
        
        return text[:pos] + new_marker + text[pos+len(old_marker):]
    
    def _extract_entities(self, text: str) -> List[str]:
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
        entities = self._extract_entities(text)
        if len(entities) < 2:
            return text
        
        pool = self.abstract_entities if use_abstract else self.real_entities
        replacements = random.sample(pool, min(len(entities), len(pool)))
        
        result = text
        for i, entity in enumerate(entities[:len(replacements)]):
            singular, plural = replacements[i]
            result = re.sub(rf'\b{entity}s\b', plural, result, flags=re.IGNORECASE)
            result = re.sub(rf'\b{entity}\b', singular, result, flags=re.IGNORECASE)
        
        return result
    
    def augment(self, text: str, use_abstract_entities: bool = False) -> str:
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
        result = self._replace_entities(text, use_abstract=True)
        if random.random() < 0.5:
            result = self._swap_premises(result)
        if random.random() < 0.5:
            result = self._replace_quantifiers(result)
        return result


def apply_augmentation(
    samples: List[Dict], 
    augmenter: SyllogismAugmenter,
    multiplier: int = 3,
    hard_case_multiplier: int = 5,
    use_debiasing_aug: bool = True,
) -> List[Dict]:
    if multiplier <= 1 and hard_case_multiplier <= 1:
        return samples
    
    augmented = []
    
    for s in samples:
        v = s.get("validity")
        p = s.get("plausibility")
        
        if v is None or p is None:
            is_hard = False
        else:
            is_hard = (v and not p) or (not v and p)
        
        actual_multiplier = hard_case_multiplier if is_hard else multiplier
        
        for k in range(actual_multiplier - 1):
            new_sample = dict(s)
            new_sample["id"] = f'{s.get("id", "")}_aug{k}'
            
            if use_debiasing_aug and is_hard:
                new_sample["syllogism"] = augmenter.augment_for_debiasing(s["syllogism"])
            else:
                new_sample["syllogism"] = augmenter.augment(s["syllogism"])
            
            augmented.append(new_sample)
    
    return samples + augmented


# =============================================================================
# Gradient Reversal Layer
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
# Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
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
# Dataset
# =============================================================================
class MultilingualSyllogismDataset(Dataset):
    """Dataset for multilingual syllogism classification"""
    
    def __init__(
        self, 
        samples, 
        tokenizer, 
        max_length=256, 
        split="train",
        use_scl=False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.use_scl = use_scl
        
        # Index by quadrant
        self.quadrant_indices = {"VP": [], "VI": [], "IP": [], "II": [], "unknown": []}
        for i, s in enumerate(samples):
            q = self._get_quadrant(s)
            self.quadrant_indices[q].append(i)
        
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
        q = self._get_quadrant(sample)
        weights = {"VP": 1.0, "II": 1.0, "VI": 2.0, "IP": 2.0, "unknown": 1.0}
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
        }
        
        # SCL pairs
        if self.split == "train" and self.use_scl:
            pos_pool = self.valid_indices if label == 1 else self.invalid_indices
            neg_pool = self.invalid_indices if label == 1 else self.valid_indices
            
            if len(pos_pool) > 1:
                pos_idx = random.choice([i for i in pos_pool if i != idx])
            else:
                pos_idx = idx
            neg_idx = random.choice(neg_pool) if neg_pool else idx
            
            pos_ids, pos_mask = self._encode(self.samples[pos_idx]["syllogism"])
            neg_ids, neg_mask = self._encode(self.samples[neg_idx]["syllogism"])
            
            item.update({
                "pos_input_ids": pos_ids,
                "pos_attention_mask": pos_mask,
                "neg_input_ids": neg_ids,
                "neg_attention_mask": neg_mask,
            })
        
        return item


# =============================================================================
# Model (supports both DeBERTa and XLM-RoBERTa)
# =============================================================================
class MultilingualClassifier(nn.Module):
    """
    Classifier supporting multiple backbone architectures:
    - XLM-RoBERTa (recommended for Subtask 3)
    - mDeBERTa
    - DeBERTa (English only)
    """
    
    def __init__(
        self,
        model_name="xlm-roberta-large",
        num_labels=2,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.15,
        lora_targets=None,  # Auto-detect if None
        use_focal_loss=False,
        focal_gamma=2.0,
        use_label_smoothing=True,
        label_smoothing=0.1,
        use_scl=True,
        scl_temp=0.07,
        scl_proj_dim=256,
        scl_weight=0.5,
        use_adversarial=False,
        adv_weight=0.3,
        grl_alpha=1.0,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_scl = use_scl
        self.scl_temp = scl_temp
        self.scl_weight = scl_weight
        self.use_adversarial = use_adversarial
        self.adv_weight = adv_weight
        self.use_focal_loss = use_focal_loss
        
        # Auto-detect LoRA targets
        if lora_targets is None:
            lora_targets = get_lora_target_modules(model_name)
        
        print(f"Model: {model_name}")
        print(f"LoRA targets: {lora_targets}")
        
        # Base model
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # LoRA
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, peft_cfg)
        self.encoder.print_trainable_parameters()
        
        h = self.encoder.config.hidden_size
        
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
        
        # Adversarial head
        if use_adversarial:
            self.grl = GradientReversalLayer(alpha=grl_alpha)
            self.plausibility_classifier = nn.Sequential(
                nn.Linear(h, h // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h // 2, 2),
            )
    
    def encode_cls(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
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
        use_rdrop=False,
        rdrop_alpha=0.7,
        pos_input_ids=None,
        pos_attention_mask=None,
        neg_input_ids=None,
        neg_attention_mask=None,
    ):
        pooled = self.encode_cls(input_ids, attention_mask)
        logits = self.classifier(pooled)
        
        loss = None
        loss_dict = {}
        
        if labels is not None:
            if self.use_focal_loss and quadrant_weight is not None:
                ce = self.criterion(logits, labels, quadrant_weight)
            else:
                ce = self.criterion(logits, labels)
            loss = ce
            loss_dict["ce_loss"] = ce.item()
            
            # R-Drop
            if use_rdrop and self.training:
                pooled2 = self.encode_cls(input_ids, attention_mask)
                logits2 = self.classifier(pooled2)
                ce2 = self.criterion(logits2, labels)
                
                kl1 = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction="batchmean")
                kl2 = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean")
                kl = 0.5 * (kl1 + kl2)
                loss = 0.5 * (ce + ce2) + rdrop_alpha * kl
                loss_dict["rdrop_kl"] = kl.item()
            
            # SCL loss
            if self.use_scl and self.training and pos_input_ids is not None:
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
            
            # Adversarial loss
            if self.use_adversarial and self.training and plausibility is not None:
                reversed_pooled = self.grl(pooled)
                plaus_logits = self.plausibility_classifier(reversed_pooled)
                adv_loss = F.cross_entropy(plaus_logits, plausibility)
                
                loss = loss + self.adv_weight * adv_loss
                loss_dict["adv_loss"] = adv_loss.item()
        
        return {"loss": loss, "logits": logits, "loss_dict": loss_dict}


# =============================================================================
# Quadrant-balanced Sampling
# =============================================================================
def build_quadrant_balanced_sampler(samples, hard_case_boost=2.0):
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
    
    total = len(samples)
    weights = []
    for q in sample_quadrants:
        if quadrant_counts[q] > 0:
            base_weight = total / (4 * quadrant_counts[q])
        else:
            base_weight = 1.0
        
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
        
        plausibility = None
        if use_adversarial:
            plausibility = batch.get("plausibility")
            if plausibility is not None:
                plausibility = plausibility.to(device)
        
        quadrant_weight = batch.get("quadrant_weight")
        if quadrant_weight is not None:
            quadrant_weight = quadrant_weight.to(device)
        
        # SCL inputs
        pos_input_ids = batch.get("pos_input_ids")
        pos_attention_mask = batch.get("pos_attention_mask")
        neg_input_ids = batch.get("neg_input_ids")
        neg_attention_mask = batch.get("neg_attention_mask")
        
        if pos_input_ids is not None:
            pos_input_ids = pos_input_ids.to(device)
            pos_attention_mask = pos_attention_mask.to(device)
            neg_input_ids = neg_input_ids.to(device)
            neg_attention_mask = neg_attention_mask.to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            plausibility=plausibility,
            quadrant_weight=quadrant_weight,
            use_rdrop=use_rdrop,
            rdrop_alpha=rdrop_alpha,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask,
        )
        
        loss = outputs["loss"] / grad_accum
        loss.backward()
        
        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += outputs["loss"].item()
        preds = outputs["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix(loss=f"{total_loss/step:.4f}", acc=f"{correct/total:.4f}")
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    # For TCE calculation
    quadrant_results = {"VP": [], "VI": [], "IP": [], "II": [], "unknown": []}
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs["loss"].item()
            preds = outputs["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    # Calculate TCE (simplified - need quadrant info for full calculation)
    pred_counts = Counter(all_preds)
    is_degenerate = min(pred_counts.values()) < total * 0.1 if len(pred_counts) > 1 else True
    
    # Approximate TCE (without quadrant info, assume balanced)
    tce_approx = 0.0
    combined_score_approx = (accuracy * 100) / (1 + math.log(1 + tce_approx)) if not is_degenerate else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "is_degenerate": is_degenerate,
        "tce_approx": tce_approx,
        "combined_score_approx": combined_score_approx,
    }


def evaluate_with_quadrants(model, dataloader, device, samples):
    """Evaluate with full quadrant information for TCE calculation"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    quadrant_correct = {"VP": 0, "VI": 0, "IP": 0, "II": 0, "unknown": 0}
    quadrant_total = {"VP": 0, "VI": 0, "IP": 0, "II": 0, "unknown": 0}
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs["loss"].item()
            preds = outputs["logits"].argmax(dim=-1)
            
            for i in range(labels.size(0)):
                if sample_idx < len(samples):
                    s = samples[sample_idx]
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
                    
                    quadrant_total[q] += 1
                    if preds[i].item() == labels[i].item():
                        quadrant_correct[q] += 1
                        correct += 1
                    total += 1
                    sample_idx += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-quadrant accuracy
    acc_VP = quadrant_correct["VP"] / quadrant_total["VP"] if quadrant_total["VP"] > 0 else 0
    acc_VI = quadrant_correct["VI"] / quadrant_total["VI"] if quadrant_total["VI"] > 0 else 0
    acc_IP = quadrant_correct["IP"] / quadrant_total["IP"] if quadrant_total["IP"] > 0 else 0
    acc_II = quadrant_correct["II"] / quadrant_total["II"] if quadrant_total["II"] > 0 else 0
    
    # TCE = |acc_plausible - acc_implausible|
    # acc_plausible = (correct_VP + correct_IP) / (total_VP + total_IP)
    # acc_implausible = (correct_VI + correct_II) / (total_VI + total_II)
    
    plausible_correct = quadrant_correct["VP"] + quadrant_correct["IP"]
    plausible_total = quadrant_total["VP"] + quadrant_total["IP"]
    implausible_correct = quadrant_correct["VI"] + quadrant_correct["II"]
    implausible_total = quadrant_total["VI"] + quadrant_total["II"]
    
    acc_plausible = plausible_correct / plausible_total if plausible_total > 0 else 0
    acc_implausible = implausible_correct / implausible_total if implausible_total > 0 else 0
    
    tce = abs(acc_plausible - acc_implausible)
    # Combined score in percentage form (matching leaderboard)
    combined_score = (accuracy * 100) / (1 + math.log(1 + tce))
    
    # Check for degenerate predictions
    min_quadrant_acc = min(acc_VP, acc_VI, acc_IP, acc_II) if all(quadrant_total[q] > 0 for q in ["VP", "VI", "IP", "II"]) else 0
    is_degenerate = min_quadrant_acc < 0.1
    
    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "acc_VP": acc_VP,
        "acc_VI": acc_VI,
        "acc_IP": acc_IP,
        "acc_II": acc_II,
        "acc_plausible": acc_plausible,
        "acc_implausible": acc_implausible,
        "tce_approx": tce,
        "combined_score_approx": combined_score,
        "min_quadrant_acc": min_quadrant_acc,
        "is_degenerate": is_degenerate,
    }


# =============================================================================
# Main Training Loop
# =============================================================================
def run_once(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model info
    model_info = get_model_info(args.model_name)
    print(f"\nModel Info:")
    print(f"  Name: {args.model_name}")
    print(f"  Languages: {model_info['languages']}")
    print(f"  Low-resource support: {model_info['low_resource']}")
    
    # Load data
    print(f"\nLoading data from: {args.official_path}")
    all_samples = load_json_list(args.official_path)
    
    if args.mode == "dev":
        train_samples, val_samples = stratified_split(all_samples, train_ratio=1-args.val_split, seed=args.seed)
        print_quadrant_distribution(train_samples, "Train")
        print_quadrant_distribution(val_samples, "Val")
    else:
        train_samples = all_samples
        val_samples = None
        print_quadrant_distribution(train_samples, "Train (full)")
    
    # Augmentation
    if args.use_augmentation:
        augmenter = SyllogismAugmenter(
            swap_premises_prob=args.swap_premises_prob,
            quantifier_replace_prob=args.quantifier_replace_prob,
            entity_replace_prob=args.entity_replace_prob,
            seed=args.seed,
        )
        train_samples = apply_augmentation(
            train_samples,
            augmenter,
            multiplier=args.aug_multiplier,
            hard_case_multiplier=args.hard_case_aug_multiplier,
            use_debiasing_aug=args.use_debiasing_aug,
        )
        print(f"\nAfter augmentation: {len(train_samples)} samples")
    
    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    train_ds = MultilingualSyllogismDataset(
        train_samples, tokenizer, args.max_length, split="train", use_scl=args.use_scl
    )
    
    if args.use_quadrant_sampling:
        sampler = build_quadrant_balanced_sampler(train_samples, hard_case_boost=args.hard_case_boost)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    val_loader = None
    if val_samples:
        val_ds = MultilingualSyllogismDataset(val_samples, tokenizer, args.max_length, split="val")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = MultilingualClassifier(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training
    best_combined = 0.0
    best_acc = 0.0
    no_improve = 0
    history = []
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*70}")
        
        use_adversarial = args.use_adversarial and epoch >= args.adv_warmup_epochs
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum=args.grad_accum,
            use_rdrop=args.use_rdrop,
            rdrop_alpha=args.rdrop_alpha,
            use_adversarial=use_adversarial,
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        if val_loader:
            metrics = evaluate_with_quadrants(model, val_loader, device, val_samples)
            print(f"Val Loss: {metrics['loss']:.4f}, Val Acc: {metrics['accuracy']:.4f}")
            print(f"  VP: {metrics['acc_VP']:.3f}, VI: {metrics['acc_VI']:.3f}, IP: {metrics['acc_IP']:.3f}, II: {metrics['acc_II']:.3f}")
            
            if metrics["is_degenerate"]:
                print(f"  ⚠️  DEGENERATE: Model predicting mostly one class!")
            else:
                print(f"  TCE (approx): {metrics['tce_approx']*100:.2f}% | Combined: {metrics['combined_score_approx']:.2f}")
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": metrics["loss"],
                "val_acc": metrics["accuracy"],
                "tce_approx": metrics["tce_approx"],
                "combined_score_approx": metrics["combined_score_approx"],
            })
            
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
                print(f"  Early stopping: {no_improve}/{args.patience}")
                if no_improve >= args.patience:
                    print("\nEarly stopping triggered.")
                    break
        else:
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})
            torch.save(model.state_dict(), f"{args.output_dir}/model_epoch{epoch+1}.pt")
    
    if args.mode == "final":
        torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
    
    # Save history and config
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
    print(f"Best combined score: {best_combined:.2f}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 70)
    
    return best_combined


def main():
    parser = argparse.ArgumentParser(description="Multilingual Syllogism Trainer for SemEval Task 11 Subtask 3")
    
    # Data
    parser.add_argument("--mode", type=str, default="dev", choices=["dev", "final"])
    parser.add_argument("--official_path", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    
    # Model - DEFAULT TO XLM-RoBERTa for multilingual
    parser.add_argument("--model_name", type=str, default="xlm-roberta-large",
                        help="Model: xlm-roberta-large (recommended), microsoft/mdeberta-v3-base, microsoft/deberta-v3-large")
    
    # Training basics
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=7)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.15)
    
    # Regularization
    parser.add_argument("--use_label_smoothing", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_rdrop", action="store_true")
    parser.add_argument("--rdrop_alpha", type=float, default=0.7)
    
    # SCL
    parser.add_argument("--use_scl", action="store_true")
    parser.add_argument("--scl_temp", type=float, default=0.07)
    parser.add_argument("--scl_proj_dim", type=int, default=256)
    parser.add_argument("--scl_weight", type=float, default=0.5)
    
    # Debiasing
    parser.add_argument("--use_adversarial", action="store_true")
    parser.add_argument("--adv_weight", type=float, default=0.2)
    parser.add_argument("--grl_alpha", type=float, default=0.5)
    parser.add_argument("--adv_warmup_epochs", type=int, default=2)
    
    parser.add_argument("--use_quadrant_sampling", action="store_true")
    parser.add_argument("--hard_case_boost", type=float, default=2.0)
    
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    # Augmentation
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--aug_multiplier", type=int, default=3)
    parser.add_argument("--hard_case_aug_multiplier", type=int, default=5)
    parser.add_argument("--use_debiasing_aug", action="store_true")
    parser.add_argument("--swap_premises_prob", type=float, default=0.3)
    parser.add_argument("--quantifier_replace_prob", type=float, default=0.3)
    parser.add_argument("--entity_replace_prob", type=float, default=0.5)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="models/subtask3_multilingual")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    
    args = parser.parse_args()
    
    # Multi-seed support
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