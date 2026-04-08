#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
encrypt.py  [Client Side]
Encrypts CSV to .bin, defaults to the 7-Protein Panel. Uses the public key 
throughout the process. The Client cannot decrypt their own output.

Usage:
  # Default: Automatically filter the 7-Protein Panel and encrypt
  python encrypt.py --input train.csv
  → train_encrypted.bin

  # Use all protein columns
  python encrypt.py --input train.csv --all-proteins

  # Specify column indices (overrides the 7-protein panel)
  python encrypt.py --input train.csv --protein-indices 3 40 50 36 83

  # Custom output filename
  python encrypt.py --input train.csv --output my_train.bin

  # Encrypt features only, without labels (pure prediction mode)
  python encrypt.py --input cohort.csv --no-labels

Input CSV format:
  Column 0     : sample ID
  Column 1     : PCR result  (Detected / Not  or  1 / 0)
  Columns 2~N  : protein expression values (column header = protein name)
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tenseal as ts
from sklearn import preprocessing

# ─────────────────────────────────────────────────────────────────────────────
# 7-Protein Panel (Fuzzy match by column name, case-insensitive)
# ─────────────────────────────────────────────────────────────────────────────
SEVEN_PANEL_NAMES = [
    'MCP-3',    # Monocyte Chemoattractant Protein-3
    'LIF-R',    # Leukemia Inhibitory Factor Receptor
    'TRANCE',   # TNF-Related Activation-Induced Cytokine
    'FGF-23',   # Fibroblast Growth Factor 23
    'NT-3',     # Neurotrophin-3
    'CXCL1',    # C-X-C Motif Chemokine Ligand 1
    'CXCL6',    # C-X-C Motif Chemokine Ligand 6
]

DEFAULT_PUBLIC_CTX = 'public_context.bin'


# ─────────────────────────────────────────────────────────────────────────────
# Load Public Key
# ─────────────────────────────────────────────────────────────────────────────
def load_public_context(path: str) -> ts.Context:
    if not os.path.isfile(path):
        sys.exit(
            f'[ERROR] Public context not found: {path!r}\n'
            f'        Download public_context.bin from GitHub.')
    with open(path, 'rb') as f:
        ctx = ts.context_from(f.read())
    if ctx.is_private():
        ctx.make_context_public()
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# 7-Protein Panel Name Matching (Fuzzy match, case-insensitive)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_panel_indices(all_columns: list, panel: list) -> tuple:
    """
    Find the column indices of the 7 proteins using fuzzy name matching.
    Returns (found_indices, found_names, missing_names)
    """
    col_lower = [c.lower().strip() for c in all_columns]
    found_indices = []
    found_names   = []
    missing_names = []

    for target in panel:
        tgt_lower = target.lower().strip()
        # Exact match priority
        if tgt_lower in col_lower:
            idx = col_lower.index(tgt_lower)
            found_indices.append(idx)
            found_names.append(all_columns[idx])
        else:
            # Fuzzy match (column name contains target)
            candidates = [i for i, c in enumerate(col_lower) if tgt_lower in c]
            if candidates:
                idx = candidates[0]  # Take the first match
                found_indices.append(idx)
                found_names.append(all_columns[idx])
                print(f'  [FUZZY] "{target}" → Matched to column "{all_columns[idx]}"')
            else:
                missing_names.append(target)

    return found_indices, found_names, missing_names


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading and Normalization
# ─────────────────────────────────────────────────────────────────────────────
def read_and_normalise(csv_path: str,
                       protein_indices: list = None,
                       all_proteins: bool = False,
                       no_labels: bool = False):
    if not os.path.isfile(csv_path):
        sys.exit(f'[ERROR] CSV not found: {csv_path!r}')

    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        sys.exit('[ERROR] CSV needs ≥ 3 columns '
                 '(sample_ID, PCR_result_placeholder, ≥1 protein).')

    sample_ids  = df.iloc[:, 0].astype(str).tolist()
    all_prot_df = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    n_total     = all_prot_df.shape[1]
    all_col_names = all_prot_df.columns.tolist()

    # Handling missing values
    n_miss = all_prot_df.isnull().sum().sum()
    if n_miss > 0:
        print(f'[WARNING] {n_miss} missing value(s) → filling with column mean.')
        all_prot_df = all_prot_df.fillna(all_prot_df.mean())

    # MinMax Normalization (apply to all first)
    scaler   = preprocessing.MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(all_prot_df.values.astype(np.float64))

    # ── Protein Column Selection Logic (Priority: index > all_proteins > 7-panel) ─────────────
    if protein_indices:
        # Explicitly specified index (overrides panel)
        bad = [i for i in protein_indices if i >= n_total]
        if bad:
            sys.exit(f'[ERROR] Indices {bad} out of range (0 ~ {n_total-1}).')
        sel_features  = features[:, protein_indices]
        used_indices  = protein_indices
        protein_names = [all_col_names[i] for i in protein_indices]
        print(f'[INFO] Mode : Custom indices ({len(protein_indices)} proteins)')

    elif all_proteins:
        # Use all proteins
        sel_features  = features
        used_indices  = list(range(n_total))
        protein_names = all_col_names
        print(f'[INFO] Mode : ALL proteins ({n_total})')

    else:
        # ✅ Default: 7-Protein Panel (Name matching)
        print(f'[INFO] Mode : 7-Protein Panel (name matching)')
        found_idx, found_names, missing = resolve_panel_indices(
            all_col_names, SEVEN_PANEL_NAMES)

        if missing:
            print(f'\n[ERROR] The following proteins could not be found in the CSV:')
            for m in missing:
                print(f'        ✗ {m}')
            print(f'\n  CSV protein columns (first 20):')
            for c in all_col_names[:20]:
                print(f'        · {c}')
            sys.exit('\n[HINT] Please check CSV column names, or use --protein-indices to specify indices manually.')

        sel_features  = features[:, found_idx]
        used_indices  = found_idx
        protein_names = found_names

    # ── Print selected proteins ─────────────────────────────────────────────────────────
    print(f'[INFO] Selected proteins ({len(protein_names)}):')
    for i, (idx, name) in enumerate(zip(used_indices, protein_names)):
        print(f'       [{i+1}] col_idx={idx}  →  {name}')

    # ── Label Loading ────────────────────────────────────────────────────────────
    labels = None
    if not no_labels:
        label_col = df.iloc[:, 1].astype(str).str.strip()
        unique    = set(label_col.unique())
        if unique.issubset({'Detected', 'Not', '0', '1'}):
            labels = label_col.map(
                {'Not': 0, 'Detected': 1,
                 '0':   0, '1':         1}).values.astype(int)
            print(f'[INFO] Labels : Detected={int(labels.sum())}, '
                  f'Not={(labels==0).sum()}, Total={len(labels)}')
        else:
            print(f'[INFO] Labels : Cannot recognize PCR column → No labels included')
    else:
        print(f'[INFO] Labels : --no-labels mode, no labels included')

    return sample_ids, sel_features, labels, used_indices, protein_names


# ─────────────────────────────────────────────────────────────────────────────
# Encryption
# ─────────────────────────────────────────────────────────────────────────────
def encrypt_vectors(ctx: ts.Context, features: np.ndarray) -> list:
    enc_list = []
    w = len(str(len(features)))
    for i, row in enumerate(features):
        enc_list.append(ts.ckks_vector(ctx, row.tolist()).serialize())
        if (i + 1) % 10 == 0 or (i + 1) == len(features):
            print(f'  [{i+1:>{w}}/{len(features)}] encrypted ...', end='\r')
    print()
    return enc_list


# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        description='PreRisk-CoV2: Encrypt CSV → encrypted .bin (7-Protein Panel)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
7-Protein Panel (Default):
  {", ".join(SEVEN_PANEL_NAMES)}

Examples:
  # Default: Automatically filter 7-Protein Panel and encrypt (with labels)
  python encrypt.py --input train.csv

  # Prediction mode (without labels)
  python encrypt.py --input new_patient.csv --no-labels

  # Use all proteins
  python encrypt.py --input train.csv --all-proteins

  # Manually specify index (overrides panel)
  python encrypt.py --input train.csv --protein-indices 3 40 50 36 83

  # Custom output filename
  python encrypt.py --input train.csv --output my_train.bin
        """)
    p.add_argument('--input',           required=True,
                   help='Input CSV path')
    p.add_argument('--output',          default=None,
                   help='Output .bin path (Default: <input>_encrypted.bin)')
    p.add_argument('--context',         default=DEFAULT_PUBLIC_CTX,
                   help=f'Public context path (Default: {DEFAULT_PUBLIC_CTX})')

    # Protein selection mode (Mutually exclusive, Priority: indices > all > panel)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument('--protein-indices', type=int, nargs='+', default=None,
                     help='Manually specify 0-based column indices, overrides 7-panel\n'
                          'e.g. --protein-indices 3 40 50 36 83')
    sel.add_argument('--all-proteins',    action='store_true',
                     help='Use all protein columns')

    p.add_argument('--no-labels',       action='store_true',
                   help='Exclude labels (pure prediction mode)')
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    if args.output is None:
        base        = os.path.splitext(os.path.basename(args.input))[0]
        out_dir     = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(out_dir, f'{base}_encrypted.bin')

    print('\n' + '='*60)
    print('  PreRisk-CoV2  |  encrypt.py  (TenSEAL CKKS)')
    print('  7-Protein Panel Mode')
    print('='*60)
    print(f'[INFO] Input     : {args.input}')
    print(f'[INFO] Context   : {args.context}')
    print(f'[INFO] Output    : {args.output}')
    print(f'[INFO] No-labels : {args.no_labels}')

    ctx = load_public_context(args.context)
    sample_ids, features, labels, used_indices, protein_names = \
        read_and_normalise(
            args.input,
            protein_indices=args.protein_indices,
            all_proteins=args.all_proteins,
            no_labels=args.no_labels,
        )

    print(f'[INFO] Shape     : {features.shape}  '
          f'({features.shape[0]} samples × {features.shape[1]} proteins)')
    print(f'\n[INFO] Encrypting ...')
    enc_list = encrypt_vectors(ctx, features)

    os.makedirs(
        os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    payload = {
        'n_samples':         len(enc_list),
        'sample_ids':        sample_ids,
        'protein_indices':   used_indices,
        'protein_names':     protein_names,
        'n_features':        len(used_indices),
        'encrypted_samples': enc_list,
        'labels':            labels.tolist() if labels is not None else None,
        'has_labels':        labels is not None,
        'panel_mode':        '7-protein' if not args.protein_indices
                             and not args.all_proteins else 'custom',
    }
    with open(args.output, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    kb = os.path.getsize(args.output) / 1024
    print(f'\n[INFO] ✅ Saved → {args.output}  ({kb:.1f} KB)')
    print(f'[INFO] ⚠️  This file is encrypted.')
    print(f'[INFO] Panel : {", ".join(protein_names)}')
    if labels is not None:
        print(f'[INFO] ✅  Label included in .bin (Visible after Server decryption)')
    else:
        print(f'[INFO] 🔒  Label not included (Pure prediction mode)')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
