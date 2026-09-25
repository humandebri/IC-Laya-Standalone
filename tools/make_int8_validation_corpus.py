#!/usr/bin/env python3
"""Build a fixed 96-input compatibility corpus using the uploaded tokenizer."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = ROOT / 'checkpoints/laya-int8/tokenizer.json'
MANIFEST = ROOT / 'checkpoints/laya-int8/manifest.json'
OUTPUT = ROOT / 'artifacts/int8_optimization_v4/validation-corpus.json'
TEXTS = [
    'I was charged twice for the same order and would like the duplicate payment refunded.',
    'The parcel arrived yesterday, but the charging cable was missing from the box.',
    'Please cancel my subscription before it renews next Monday.',
    'My account is locked after changing phones. I still have access to my email.',
    'The invoice lists a different company name and an incorrect tax address.',
    'Our team needs a copy of the receipt for the March payment.',
    'The delivery status says complete, but nobody at the reception desk received it.',
    'I ordered the blue version and received a red one. Can you exchange it?',
    'The app crashes each time I upload a photo from my tablet.',
    'We have not received the confirmation email after resetting the password.',
    'I want to upgrade from the monthly plan to the annual plan this week.',
    'The discount code was accepted, but the final charge used the full price.',
    'Please remove the old administrator from our organization account.',
    'The replacement device stopped working after two days of normal use.',
    'I accidentally placed the same order twice and only need one item.',
    'Can you explain why the estimated shipping date moved back by five days?',
    'The report export contains blank rows even though the dashboard shows data.',
    'Our finance team needs the invoice currency changed for future purchases.',
    'I would like a human to review this decision because the amount seems wrong.',
    'The security alert shows a login from a location I do not recognize.',
    'Please confirm whether the transfer completed before I submit it again.',
    'The product description promised a warranty, but the support page says otherwise.',
    'Our event starts tomorrow and the tickets have not appeared in the account.',
    'I appreciate the quick response. The issue is resolved and no further action is needed.',
]
LENGTHS = (35, 63, 64, 65, 96, 112, 127, 128)


def main():
    raw = TOKENIZER.read_bytes()
    manifest = json.loads(MANIFEST.read_text())
    digest = hashlib.sha256(raw).digest()
    if list(digest) != manifest['tokenizer_sha256']:
        raise RuntimeError('tokenizer differs from the uploaded pack')
    tok = Tokenizer.from_file(str(TOKENIZER))
    cases = []
    for name in ('choice', 'noul', 'score'):
        source = json.loads((ROOT / f'artifacts/laya-{name}-input.json').read_text())
        ids = source['input_ids']
        end = ids.index(50282, source['markers'][-1]) + 1
        prefix, terminator = ids[:end], ids[-1]
        if terminator != 50282:
            raise RuntimeError('unexpected template terminator')
        for index, text in enumerate(TEXTS):
            body = tok.encode(text, add_special_tokens=False).ids
            input_ids = prefix + body[:128 - end - 1] + [terminator]
            cases.append({'id': f'{name}-natural-{index:02d}', 'schema': name,
                          'kind': 'natural_text', 'text': text,
                          'input': {'input_ids': input_ids, 'markers': source['markers'],
                                    'qtype_id': source['qtype_id']}})
        body = tok.encode(TEXTS[0], add_special_tokens=False).ids
        for length in LENGTHS:
            count = length - end - 1
            if count < 1:
                raise RuntimeError(f'{name}: {length} is shorter than the prefix')
            input_ids = prefix + (body * ((count + len(body) - 1) // len(body)))[:count] + [terminator]
            cases.append({'id': f'{name}-boundary-{length}', 'schema': name,
                          'kind': 'length_boundary',
                          'input': {'input_ids': input_ids, 'markers': source['markers'],
                                    'qtype_id': source['qtype_id']}})
    if len(cases) != 96 or len({c['id'] for c in cases}) != 96:
        raise RuntimeError('unexpected corpus size')
    report = {'tokenizer_sha256': digest.hex(), 'tokenizers_version': version('tokenizers'),
              'pack_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              'cases': cases, 'note': 'Compatibility inputs, not labeled accuracy data.'}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + '\n')
    print(OUTPUT, len(cases))


if __name__ == '__main__':
    main()
