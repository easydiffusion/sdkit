from typing import List, Tuple, Union

import regex as re

from sdkit.utils import log

ROUND_PRECISION = 3
DELIMITERS = ["()", "[]", "`", '"']


def clean_text(text: str) -> str:
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def clean_transforms(transforms: Union[dict, list], parent_text: str) -> Union[dict, list]:
    if transforms is None:
        return None
    if isinstance(transforms, list):
        newList = []
        for tr in transforms:
            tr = clean_transforms(tr, parent_text)
            # TODO Merge duplicates together.
            if not tr:
                continue
            # Remove useless operations and useless nesting.
            if tr["text"] == parent_text and "slerp" in tr:
                if tr["slerp"] < 0:
                    # A negative slerp on self is in fact just an inverse.
                    # Replace by negative weight to apply negative part properly.
                    del tr["slerp"]
                    tr["weight"] = -1.0
                    newList.append(tr)
                elif "transforms" in tr and tr["transforms"] is not None:
                    assert isinstance(tr["transforms"], list), "Subtransforms isnt a list! tr[transforms] %s" % tr
                    newList.extend(tr["transforms"])
                continue
            if len(tr) == 2 and "transforms" in tr and not any(["slerp" in st for st in tr["transforms"]]):
                if tr["transforms"] is not None:
                    assert isinstance(tr["transforms"], list), "Subtransforms isnt a list! tr[transforms] %s" % tr
                    newList.extend(tr["transforms"])
                continue
            newList.append(tr)
        if len(newList) > 0:
            return newList
        return None
    if isinstance(transforms, dict):
        if "text" not in transforms:
            log.warn("Invalid transform! Missing text field to apply transform. Removed from list.", transforms)
            return None  # Cant apply a transform without a target.
        if "weight" in transforms and round(transforms["weight"], ROUND_PRECISION) == 1.0:
            del transforms["weight"]
        if "slerp" in transforms and round(transforms["slerp"], ROUND_PRECISION) == 0.0:
            del transforms["slerp"]
        if "slerp" not in transforms and "transforms" in transforms:
            has_reduced = True
            while has_reduced and "transforms" in transforms:
                has_reduced = False
                subtransforms = clean_transforms(transforms["transforms"], transforms["text"])
                if (
                    subtransforms
                    and len(subtransforms) == 1
                    and "text" in subtransforms[0]
                    and subtransforms[0]["text"] == transforms["text"]
                    and "weight" in subtransforms[0]
                ):
                    if "weight" in transforms:
                        transforms["weight"] = transforms["weight"] * subtransforms[0]["weight"]
                    else:
                        transforms["weight"] = subtransforms[0]["weight"]
                    if round(transforms["weight"], ROUND_PRECISION) == 1.0:
                        del transforms["weight"]
                    if "transforms" in subtransforms[0]:
                        subtransforms = subtransforms[0]["transforms"]
                        has_reduced = True
                    else:
                        subtransforms = None
                if subtransforms is None or len(subtransforms) == 0:
                    del transforms["transforms"]
                else:
                    transforms["transforms"] = subtransforms
        elif "transforms" in transforms:
            transforms["transforms"] = clean_transforms(transforms["transforms"], transforms["text"])
        if "transforms" in transforms and (transforms["transforms"] is None or len(transforms["transforms"]) == 0):
            del transforms["transforms"]
        if (
            len(transforms) == 2
            and "text" in transforms
            and "transforms" in transforms
            and len(transforms["transforms"]) == 1
            and "slerp" not in transforms["transforms"][0]
        ):
            return transforms["transforms"][0]
        if len(transforms) == 1 and "text" in transforms:
            return None
        return transforms
    raise ValueError("transforms must be a list or dict.")


def parse_prompt(prompt: str) -> Tuple[str, dict]:
    """
    Parse a prompt to produce a series of transforms to be applied on the conditionings.
    Requires model to be on the device
    """
    transforms = []
    prompt_cleaned = ""
    for delimiter, quoted_prompt in split_quotes(prompt, DELIMITERS):
        if delimiter == "()":
            level = 1
        elif delimiter == "[]":
            level = -1
        else:
            level = 0

        if not quoted_prompt:
            prompt_cleaned += " " + delimiter
            if level != 0:
                transforms.append({"text": delimiter, "weight": 1.1**level})
            continue

        if quoted_prompt.startswith(":") and len(transforms) > 0:
            prompt_cleaned += quoted_prompt
            # Split at first space
            if " " in quoted_prompt:
                space_idx = quoted_prompt.index(" ")
                transforms[-1] = {
                    "text": transforms[-1]["text"] + quoted_prompt[:space_idx],
                    "weight": 1.1**level,
                    "transforms": [transforms[-1]],
                }
                quoted_prompt = quoted_prompt[space_idx:]
            else:
                transforms[-1] = {
                    "text": transforms[-1]["text"] + quoted_prompt,
                    "weight": 1.1**level,
                    "transforms": [transforms[-1]],
                }
                continue

        subtransforms = None
        subprompt = quoted_prompt
        if quoted_prompt != prompt and any([d[0] in quoted_prompt for d in DELIMITERS]):
            subprompt, subtransforms = parse_prompt(quoted_prompt)
        elif delimiter:
            subprompt, subtransforms = parse_segment(quoted_prompt)
        if quoted_prompt != subprompt:
            prompt_cleaned += " " + subprompt
            transforms.append({"text": subprompt, "weight": 1.1**level, "transforms": subtransforms})
            continue
        if quoted_prompt:
            prompt_cleaned += " " + quoted_prompt
        for subprompt in re.findall(r"(?:[^\(\)\[\]\"\:]+)+:?-?\d*[.,]?\d*", quoted_prompt):
            # Can be only empty space.
            subprompt = subprompt.strip()
            if subprompt:  # Will be empty if was all spaces.
                transforms.append({"text": subprompt, "weight": 1.1**level})
    if not any([":" in t["text"] for t in transforms]):
        return clean_text(prompt_cleaned), clean_transforms(transforms, prompt_cleaned)

    weighted_subprompt = " ".join([tr["text"] for tr in transforms])
    subprompt, subtransforms = parse_segment(weighted_subprompt)
    if not subtransforms:
        subprompts, weights = [list(t) for t in zip(*split_weighted_subprompts(weighted_subprompt))]
        return subprompts[0], [{"text": subprompts[0], "weight": weights[0]}]

    assert len(transforms) >= len(subtransforms), "Too many subtransforms %s" % (len(subtransforms) - len(transforms))
    transforms_bundle = []
    weighted_transforms = []
    while len(transforms) > 0:
        tr = transforms.pop(0)
        transforms_bundle.append(tr)
        tr_text = tr["text"]
        if ":" not in tr_text:
            continue
        tr_text = tr_text[0 : tr_text.rindex(":")]
        tr["text"] = tr_text
        st = subtransforms.pop(0)
        assert "transforms" not in st or st["transforms"] is None, "subtransforms already defined %s" % st["transforms"]
        st["transforms"] = transforms_bundle
        transforms_bundle = []
        weighted_transforms.append(st)
    if transforms_bundle or subtransforms:
        assert len(subtransforms) == 1, "subtransforms left %s" % (len(subtransforms))
        st = subtransforms[0]
        assert "transforms" not in st or st["transforms"] is None, "subtransforms already defined %s" % st["transforms"]
        if transforms_bundle:
            st["transforms"] = transforms_bundle
        weighted_transforms.append(st)
    transforms = weighted_transforms
    return subprompt, clean_transforms(transforms, prompt_cleaned)


def parse_segment(text: str) -> Tuple[str, dict]:
    subprompts, weights = [list(t) for t in zip(*split_weighted_subprompts(text))]
    if len(subprompts) == 0:
        return "", None
    elif len(subprompts) == 1:
        if round(weights[0], ROUND_PRECISION) == 1.0:
            return f"{subprompts[0]}", None
        else:
            return subprompts[0], [{"text": subprompts[0], "weight": weights[0]}]
    else:
        transforms = []
        # Count the sum as all positive terms to scale, but keep sign for later.
        weights_sum = sum([abs(w) for w in weights])
        for i, subprompt in enumerate(subprompts):
            subprompt, subtransforms = parse_prompt(subprompt)
            if i == 0:
                subprompts[0] = subprompt
            if weights_sum != 0:
                weights[i] /= weights_sum
            if "(" in subprompt:
                log.warn("Mismatched ()")
            if "[" in subprompt:
                log.warn("Mismatched []")
            if round(weights[i], ROUND_PRECISION) != 0.0:
                transforms.append({"text": subprompt, "slerp": weights[i], "transforms": subtransforms})
        return subprompts[0], transforms


def split_quotes(stringToSplit: str, delimiters: List[str]) -> List[Tuple[str, str]]:
    """
    Splits the string passed in by the delimiters passed in.
    Quoted sections are not split, and all tokens have whitespace trimmed from the start and end.
    returns: The quotes char, the contained text and the nesting level as a tuple.
    """
    assert isinstance(stringToSplit, str)
    if isinstance(delimiters, str):
        delimiters = [char for char in delimiters]
    assert isinstance(delimiters, list)
    for char in delimiters:
        if isinstance(char, str):
            assert len(char) <= 2, "Splitter char or pair of chars only"
            continue
        assert isinstance(char, tuple)
        assert len(char) == 2  # ('[', ']') ('{', '}') ('(', ')')
        assert all(isinstance(c, str) and len(c) == 1 for c in char)

    if len(stringToSplit) == 0:
        return

    quoteChar = []
    currentToken = []
    for currentCharacter in stringToSplit:
        # Compare quoteChar[0] to currentCharacter to only disable single quote chars.
        # Those should never nest as it's the same char to mark start/end of quote.
        if (len(quoteChar) <= 0 or quoteChar[0] != currentCharacter) and any(
            [currentCharacter == d[0] for d in delimiters]
        ):
            # Start of quote.
            if len(quoteChar) <= 0:
                result = "".join(currentToken).strip()
                if len(result) > 0:
                    yield None, result
                currentToken = []
            else:
                currentToken.append(currentCharacter)
            for d in delimiters:
                if currentCharacter == d[0]:
                    quoteChar.insert(0, d)
                    break
            else:
                quoteChar.insert(0, currentCharacter)
        elif len(quoteChar) > 0 and currentCharacter == quoteChar[0][-1]:
            # End quote.
            state = "".join(quoteChar[0])
            quoteChar.pop(0)
            if len(quoteChar) == 0:
                yield state, "".join(currentToken).strip()
                currentToken = []
            else:
                currentToken.append(currentCharacter)
        else:
            currentToken.append(currentCharacter)

    lastResult = "".join(currentToken).strip()
    if lastResult:
        if quoteChar:
            # Found an unmatched char and was removed, resume parsing.
            first = True
            for delim_state, substring in split_quotes(lastResult, delimiters):
                if first:
                    first = False
                    # Add first unmatched delimiter char to string result.
                    # other chars are part of currentToken
                    yield delim_state, (quoteChar[-1][0] + substring).strip()
                else:
                    # Continue parsing.
                    yield delim_state, substring
        else:
            # No quotes left to parse, return the last part of the string.
            yield None, lastResult


def split_weighted_subprompts(text: str) -> List[Tuple[str, float]]:
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    while remaining > 0:
        if ":" in text:
            idx = text.index(":")  # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx + 1 :]
            # find value for weight
            if " " in text:
                idx = text.index(" ")  # first occurence
            else:  # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except:  # couldn't treat as float
                    log.warn(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else:  # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx + 1 :]
            # append the sub-prompt and its weight
            yield prompt, weight
        else:  # no : found
            if len(text) > 0:  # there is still text though
                # take remainder as weight 1
                yield text, 1.0
            remaining = 0
