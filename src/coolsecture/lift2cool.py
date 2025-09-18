#!/usr/bin/env python3
import argparse
from .post_common import (
    infer_resolution_from_liftover, ChromIndexingFAI, read_contacts, read_chr_sizes_from_fai,
    _build_bins_from_sizes, _pixels_from_contacts, _write_cool_minimal
)

def main():
    p = argparse.ArgumentParser(description="Create two Cooler matrices (Observed.cool and Control.cool) in the genome")
    p.add_argument("--liftover", required=True, help="Path to .liftContacts file")
    p.add_argument("--fadix", required=True, help="Path to FASTA index (.fai) file")
    p.add_argument("--assembly", help="Value for Cooler 'assembly' metadata (e.g., 'TAIR10'); optional")
    p.add_argument("--out-prefix", required=True, help="Output prefix")
    args = p.parse_args()

    res = infer_resolution_from_liftover(args.liftover)
    Order = ChromIndexingFAI(args.fadix)
    contacts = read_contacts(args.liftover, Order, res, short=False)
    sizes_bp = read_chr_sizes_from_fai(args.fadix)
    bins_df  = _build_bins_from_sizes(sizes_bp, res)
    px_obs   = _pixels_from_contacts(contacts, res, sizes_bp, which='observed')
    px_ctl   = _pixels_from_contacts(contacts, res, sizes_bp, which='control')

    uri_obs = f"{args.out_prefix}.Observed.cool"
    uri_ctl = f"{args.out_prefix}.Control.cool"
    _write_cool_minimal(uri_obs, bins_df, px_obs, assembly=args.assembly)
    _write_cool_minimal(uri_ctl, bins_df, px_ctl, assembly=args.assembly)
    print(f"[OK] wrote {uri_obs}")
    print(f"[OK] wrote {uri_ctl}")
    print("Note: COOL matrices remain in source coordinates; values represent 0–100 percentiles.")

if __name__ == "__main__":
    main()
