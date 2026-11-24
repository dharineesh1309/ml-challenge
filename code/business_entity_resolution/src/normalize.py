"""Deterministic record normalization -> several representations per record.

Raw fields are kept next to every derived one so errors can be traced.
Vectorised with polars; only rows containing non-ASCII text go through
Python (anyascii transliteration: Indic scripts, accents, curly quotes).

Tables are intentionally small and country-agnostic: an unknown token simply
passes through unchanged, so new countries (France in test) still work.
"""
import polars as pl
from anyascii import anyascii

from data_io import cached, read_source

VERSION = "n4"  # bump when any rule below changes (invalidates caches)

# OCR-style digit-for-letter swaps inside mixed tokens (f0x, 5ervices, dou6las, t0lia)
OCR_FROM, OCR_TO = list("01345678"), list("oleastgb")

# Legal-form class. Same name + same address + DIFFERENT legal form is a deliberate hard
# negative in this data (Swastik It Pvt Ltd vs Swastik It LLP), so the class is kept as a
# feature even though the tokens are stripped from the name. First match wins.
LEGAL_CLASSES = [
    ("pvt_ltd", {"pvt", "private", "praivet", "praibhet", "piraivet", "praivrr", "pra"}),
    ("pub_ltd", {"public"}),
    ("llp", {"llp", "elelpi"}), ("pllc", {"pllc"}), ("llc", {"llc"}), ("lp", {"lp"}),
    ("pc", {"pc"}), ("pa", {"pa"}), ("plc", {"plc"}), ("inc", {"inc", "incorporated", "lnc"}),
    ("corp", {"corp", "corporation"}),
    ("ltd", {"ltd", "limited", "limitet", "limirrd", "limitedd", "li"}),
    ("and_co", None),                     # "X & Co": its own class (sources swap it freely)
    ("co", {"co", "company"}),
    ("sarl", {"sarl"}), ("sasu", {"sasu"}), ("sas", {"sas"}), ("eurl", {"eurl"}), ("sa", {"sa"}),
    ("sci", {"sci"}), ("snc", {"snc"}), ("gmbh", {"gmbh"}),
]
_LTD = {"ltd", "limited", "limitet", "limirrd", "limitedd", "li"}

LEGAL = {
    "inc", "incorporated", "lnc", "llc", "llp", "lp", "ltd", "limited", "pvt", "private", "corp",
    "corporation", "co", "company", "pc", "pllc", "plc", "pa", "the", "and", "of", "dba",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "cie", "gmbh",
    # anyascii renderings of Indic legal words seen in blocking misses
    "praivet", "praibhet", "piraivet", "praivrr", "pra", "li", "elelpi", "limitet", "limirrd", "limitedd",
}
# Words the sources add/drop most often between true pairs (measured on 964k train pairs):
# removed from the name *signature* only; name_core keeps them for features.
GENERIC = {
    "center", "centre", "services", "service", "partners", "group", "associates", "care", "clinic",
    "solutions", "trading", "india", "shri", "sri", "smt", "mr", "mrs", "ms", "dr", "formerly",
    "aka", "as", "doing", "business", "id",
}

# token -> canonical token (both spellings collapse to one form)
ADDR_MAP = {
    "street": "st", "str": "st", "saint": "st",
    "avenue": "ave", "av": "ave", "avn": "ave",
    "road": "rd", "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct",
    "boulevard": "blvd", "bd": "blvd", "blv": "blvd", "place": "pl", "highway": "hwy",
    "parkway": "pkwy", "circle": "cir", "terrace": "ter", "trail": "trl", "square": "sq",
    "route": "rte", "mount": "mt", "fort": "ft", "heights": "hts", "point": "pt",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "r": "rue", "allee": "all", "impasse": "imp", "chemin": "chem", "faubourg": "fbg",
    "ngr": "nagar", "col": "colony", "sec": "sector",
    # ordinal words -> numbers (12th Street vs Twelfth St)
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11", "twelfth": "12",
    "thirteenth": "13", "fourteenth": "14", "fifteenth": "15", "sixteenth": "16",
    "seventeenth": "17", "eighteenth": "18", "nineteenth": "19", "twentieth": "20",
    "thirtieth": "30", "fortieth": "40", "fiftieth": "50",
    # single-word US / Indian state names -> the codes other sources use
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    "karnataka": "ka", "maharashtra": "mh", "gujarat": "gj", "rajasthan": "rj", "kerala": "kl",
    "telangana": "tg", "ts": "tg", "haryana": "hr", "punjab": "pb", "bihar": "br", "odisha": "od",
    "orissa": "od", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg",
    "uttarakhand": "uk",
}
# multi-word state names, replaced on the raw address string before tokenizing
MULTI_STATES = {
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "west virginia": "wv", "district of columbia": "dc",
    "tamil nadu": "tn", "uttar pradesh": "up", "madhya pradesh": "mp", "andhra pradesh": "ap",
    "west bengal": "wb", "himachal pradesh": "hp", "jammu and kashmir": "jk", "jammu kashmir": "jk",
}
# tokens that carry no locating information
ADDR_DROP = {
    "unit", "apt", "apartment", "suite", "ste", "fl", "floor", "po", "box", "no", "near", "nr",
    "opp", "opposite", "behind", "hno", "flat", "door", "plot", "shop", "bldg", "building", "null",
}

NON_LATIN = r"[Ͱ-῿Ⰰ-퟿豈-￯]"
DOMAIN = r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:co\.in|com|net|org|in|fr|biz|info|us|co)\b"


def skeleton(x: pl.Expr) -> pl.Expr:
    """Consonant skeleton per token: robust to vowel typos and to how anyascii spells
    Indic scripts (Global/globl -> glbl, Indian/imdiyn -> indn, Exports/eksports -> eksprts)."""
    x = (x.str.replace_all("ph", "f").str.replace_all("x", "ks").str.replace_all("[cq]", "k")
         .str.replace_all("z", "s").str.replace_all("v", "w")
         .str.replace_all(r"m([dfgjklnrstw])", "n$1")        # anusvara transliterated as m
         .str.replace_all(r"\B[aeiouyhw]", ""))              # drop non-initial vowels/semivowels
    for ch in "bdfgjklmnprstw":
        x = x.str.replace_all(ch + "+", ch)                  # collapse doubled consonants
    return x


def fold(s: pl.Series) -> pl.Series:
    """ASCII-fold only the rows that need it (anyascii is Python-speed)."""
    mask = s.str.contains(r"[^\x00-\x7F]")
    if not mask.any():
        return s
    return s.clone().scatter(mask.arg_true(), [anyascii(x) for x in s.filter(mask).to_list()])


def normalize_records(df: pl.DataFrame, src: int) -> pl.DataFrame:
    name = fold(df["business_name"]).str.to_lowercase()
    addr = fold(df["business_address"]).str.to_lowercase()
    out = df.select(
        entity_id="entity_id", src=pl.lit(src, pl.UInt8), country="country",
        name_raw="business_name", addr_raw="business_address",
        name_nonlatin=pl.col("business_name").str.contains(NON_LATIN),
        addr_nonlatin=pl.col("business_address").str.contains(NON_LATIN),
    ).with_columns(_n=name, _a=addr)

    name_norm = (pl.col("_n").str.replace_all("&", " and ")
                 .str.replace_all(DOMAIN, "$1")
                 .str.replace_all(r"[.'`]", "")          # l.l.c -> llc, orelee's -> orelees
                 .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    el = pl.element()
    ocr = pl.when(el.str.contains("[a-z]") & el.str.contains("[0-9]")).then(
        el.str.replace_many(OCR_FROM, OCR_TO)).otherwise(el)
    out = out.with_columns(name_norm=name_norm.str.split(" ").list.eval(ocr).list.join(" "))
    ntok = pl.col("name_norm").str.split(" ")
    has = lambda words: ntok.list.eval(pl.element().is_in(list(words))).list.any()
    legal = pl.lit(None, pl.Utf8)
    for cls, words in reversed(LEGAL_CLASSES):      # reversed so the first class in the list wins
        if cls == "and_co":
            cond = pl.col("name_norm").str.contains(r"\band (?:co|company)$")
        elif cls in ("pvt_ltd", "pub_ltd"):
            cond = has(words) & has(_LTD)
        else:
            cond = has(words)
        legal = pl.when(cond).then(pl.lit(cls)).otherwise(legal)
    out = out.with_columns(legal_form=legal.fill_null(""))
    core = (pl.col("name_norm").str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(LEGAL)))
            .list.unique(maintain_order=True))            # "chavez chavez" -> "chavez"
    sig = pl.col("_core").list.eval(pl.element().filter(
        ~pl.element().is_in(GENERIC) & (pl.element().str.len_chars() >= 2)))
    out = out.with_columns(_core=core).with_columns(
        _sig=pl.when(sig.list.len() > 0).then(sig).otherwise(pl.col("_core")))
    out = out.with_columns(
        name_core=pl.col("_core").list.join(" "),
        name_sig=pl.col("_sig").list.join(" "),           # distinctive tokens only
        name_cat=pl.col("_sig").list.join(""),
    ).with_columns(name_skel=skeleton(pl.col("name_sig")))

    addr_norm = (pl.col("_a").str.replace_all(r"<null>", " ")
                 .str.replace_all(r"\b([a-z])\.", "$1")                # c.i.t. -> cit
                 .str.replace_all(r"(\d+)(?:st|nd|rd|th)\b", "$1")   # 2nd -> 2
                 .str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars())
    for full, code in MULTI_STATES.items():
        addr_norm = addr_norm.str.replace_all(rf"\b{full}\b", code)
    out = out.with_columns(addr_norm=addr_norm)
    toks = pl.col("addr_norm").str.split(" ").list.eval(pl.element().replace(ADDR_MAP))
    out = out.with_columns(_m=toks.list.join(" ")).with_columns(
        addr_words=toks.list.eval(pl.element().filter(
            pl.element().str.contains("[a-z]") & ~pl.element().is_in(ADDR_DROP))),
        addr_nums=pl.col("_m").str.extract_all(r"\d+")          # after ordinal words -> digits
        .list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != "")),
    ).with_columns(
        addr_norm=pl.col("addr_words").list.join(" "),     # canonical-token address string
        house_no=pl.col("addr_nums").list.first().fill_null(""),
    )
    return out.drop("_n", "_a", "_core", "_sig", "_m")


def normalized(cfg, split, src):
    """Cached normalized table for one source file of one split."""
    path = cfg.path("work_dir") / "cache" / f"norm_{VERSION}_{split}_s{src}.parquet"
    return cached(path, lambda: normalize_records(read_source(cfg.path("data_dir"), split, src), src))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    demo = pl.DataFrame({
        "entity_id": ["S2-1", "S3-2", "S2-3", "S3-4"],
        "business_name": ["-- Holloway Peak Inc Seafood", "maurewilliamscolombier.com",
                          "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "Chavez Chavez [LLC] Léarning"],
        "business_address": ["##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri", "",
                             "6(29), C.I.T. COLONY, 2ND MAIN ROAD, CHENNAI", "63 R. DE DIEPPE, LILLE"],
        "country": ["US", "US", "India", "France"],
    })
    r = normalize_records(demo, 2)
    with pl.Config(fmt_str_lengths=60, tbl_cols=-1, tbl_width_chars=250):
        print(r.select("name_core", "name_cat", "addr_words", "addr_nums", "house_no", "name_nonlatin"))
    assert r["name_core"].to_list()[0] == "holloway peak seafood"
    assert r["name_cat"].to_list()[1] == "maurewilliamscolombier"
    assert r["house_no"].to_list() == ["8", "", "6", "63"]
    assert r["addr_words"].to_list()[3] == ["rue", "de", "dieppe", "lille"]
    assert r["name_nonlatin"].to_list() == [False, False, True, False]
    assert r["name_core"].to_list()[3] == "chavez learning"
    lg = normalize_records(pl.DataFrame({
        "entity_id": ["S1-1", "S2-2", "S2-3", "S3-4", "S3-5"],
        "business_name": ["Swastik It Private Limited", "Swastik It Llp", "Mumbai Sree Limited",
                          "Boclaris L.L.C.", "Grand Classic Transition"],
        "business_address": ["223 Twelfth St, Rio Salado, Arizona", "AZ, 223 12th Street", "Tamil Nadu",
                             "North Carolina", ""],
        "country": ["US"] * 5}), 1)
    assert lg["legal_form"].to_list() == ["pvt_ltd", "llp", "ltd", "llc", ""], lg["legal_form"].to_list()
    oc = normalize_records(pl.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3"],
                                         "business_name": ["Inc F0x & Rizu", "Desert Program & Co", "#6 Rio PLLC"],
                                         "business_address": ["", "", ""], "country": ["US"] * 3}), 2)
    assert oc["name_core"].to_list() == ["fox rizu", "desert program", "6 rio"], oc["name_core"].to_list()
    assert oc["legal_form"].to_list() == ["inc", "and_co", "pllc"], oc["legal_form"].to_list()
    assert lg["addr_nums"].to_list()[:2] == [["223", "12"], ["223", "12"]], lg["addr_nums"].to_list()
    assert "az" in lg["addr_words"].to_list()[0] and lg["addr_words"].to_list()[2:4] == [["tn"], ["nc"]]
    sk = pl.select(skeleton(pl.lit("global imtrnesnl eksports ventures modrn"))).item()
    assert sk == "glbl intrnsnl eksprts wntrs mdrn", sk
    assert r["addr_nums"].to_list()[0] == ["8"]
    assert "cit" in r["addr_words"].to_list()[2]
    print("normalize self-check OK")
