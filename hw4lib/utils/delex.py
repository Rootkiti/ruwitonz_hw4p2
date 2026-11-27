"""Delexicalization helpers for WebNLG SFT: map entities/numbers to ENT#/NUM# tags and back."""
import re

_NUM_RE = re.compile(r'^[\d][\d\s.,:/\-]*$')


def _parse_triple(triple):
    parts = [p.strip() for p in triple.split("|")]
    return parts if len(parts) == 3 else None


def _surface(entity):
    return entity.strip().strip('"').replace("_", " ")


def build_delex_map(triples):
    """Assign ENT#/NUM# placeholders to each distinct subject/object."""
    raw_to_tag, surface_to_tag, tag_to_surface = {}, {}, {}
    n_ent = n_num = 0
    for triple in triples:
        parsed = _parse_triple(triple)
        if parsed is None:
            continue
        for raw in (parsed[0], parsed[2]):
            if raw in raw_to_tag:
                continue
            surface = _surface(raw)
            if _NUM_RE.match(surface):
                n_num += 1
                tag = f"NUM{n_num}"
            else:
                n_ent += 1
                tag = f"ENT{n_ent}"
            raw_to_tag[raw] = tag
            surface_to_tag[surface] = tag
            tag_to_surface[tag] = surface
    return raw_to_tag, surface_to_tag, tag_to_surface


def delexicalize_triples(triples, raw_to_tag):
    out = []
    for triple in triples:
        for raw in sorted(raw_to_tag, key=len, reverse=True):
            triple = triple.replace(raw, raw_to_tag[raw])
        out.append(triple)
    return out


def delexicalize_text(text, surface_to_tag):
    for surface in sorted(surface_to_tag, key=len, reverse=True):
        text = text.replace(surface, surface_to_tag[surface])
    return text


def relexicalize_text(text, tag_to_surface):
    for tag in sorted(tag_to_surface, key=len, reverse=True):
        text = text.replace(tag, tag_to_surface[tag])
    return text