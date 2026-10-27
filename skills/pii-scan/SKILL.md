---
name: pii-scan
description: Scan a dataset for personal data before it leaves its secure home — government IDs (SSN, passport), payment cards, bank accounts, emails, phone numbers, IP addresses, names, addresses, birth dates, precise locations and sensitive attributes, including PII hidden in free-text notes. Use whenever exporting, sharing, publishing, uploading or sampling data, sending rows to an LLM or external API, creating test fixtures or demo data, writing data to logs, or committing data files to a repository.
---

# PII scan

Personal data leaks through the side door: a `notes` column with customers' phone numbers, a card number pasted into a support ticket, a "sample" CSV committed to a public repo. Scan before anything leaves its controlled environment.

## Steps

1. **Ask where the data is going** (a teammate, a vendor, a public repo, an LLM prompt, a dashboard) and what the purpose needs. Most purposes need far fewer columns than the export has.

2. **Run the scan**:

   ```bash
   python <skill-dir>/scripts/check_pii.py --data <file> [--sample 20000]
   ```

   `<skill-dir>` is the folder that contains this SKILL.md. It reads CSV, TSV, Parquet, JSON and JSONL, checks column names and values, and prints masked examples only. Exit code 1 means a FAIL. Add `--json` for structured output.

3. **Act on the findings**:

   | Finding | What to do |
   | --- | --- |
   | `ssn`, `government_id`, `credit_card`, `iban`, `financial_account` FAIL | Remove the column. If it is truly needed, the data stays inside a controlled environment and does not get shared |
   | `free_text` | Redact matches (replace with `[EMAIL]`, `[PHONE]`), or drop the column; free text cannot be fully cleaned by patterns |
   | `email`, `phone`, `person_name`, `address`, `ip_address` | Drop, or replace with a salted hash or a surrogate key when rows must stay linkable |
   | `birth_date`, `precise_location` | Generalise (birth year or age band; round coordinates, use region) |
   | `sensitive_attribute` | Needs an explicit legal basis; drop unless the purpose requires it |

4. **Re-scan the cleaned file** and confirm PASS (or only WARNs that were accepted on purpose).

5. **Remember what the scan cannot see**: names inside free text, rare identifiers (employee numbers, medical record numbers), and re-identification through combinations such as ZIP code + birth date + gender. For anything published, aggregate or apply k-anonymity rather than relying on column removal.

## Output

Where the data is going, the columns removed, hashed or generalised, the re-scan verdict, and any remaining risk in one sentence. Never paste raw PII values into chat, logs, commit messages or prompts while doing this.
