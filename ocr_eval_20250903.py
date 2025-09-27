#!/usr/bin/env python3
"""
OCR Quality Evaluation Tool

Evaluates OCR technology quality by measuring character recognition errors
while being robust to layout recognition issues such as line merging, splitting,
and reordering.

This script compares a ground truth text file with an OCR output file,
calculates an error score, and provides detailed matching statistics.
"""

__version__ = "1.5.1"

import argparse
import sys
import time
import unicodedata
import re
from typing import List, Tuple, Dict, Set, Optional, Callable
from dataclasses import dataclass

# --- Data Structures ---

@dataclass
class MatchCandidate:
    """
    Represents a potential match between original and OCR lines.

    Attributes:
        original_indices: List of indices of the line(s) from the original text.
        ocr_indices: List of indices of the line(s) from the OCR text.
        match_type: The type of match, e.g., "1-to-1", "N-to-1", "1-to-N".
        error_score: The Levenshtein distance for this match.
        original_text_length: Total character count of matched original text.
        ocr_text_length: Total character count of matched OCR text.
    """
    original_indices: List[int]
    ocr_indices: List[int]
    match_type: str
    error_score: int
    original_text_length: int
    ocr_text_length: int

@dataclass
class SubstringMatch:
    """
    Represents an approximate substring match found during the discovery phase.

    Attributes:
        secondary_index: The index of the line in the secondary list that matched.
        start_pos: The start position of the match in the primary text.
        end_pos: The end position of the match in the primary text.
        error_score: The Levenshtein distance of the substring match.
    """
    secondary_index: int
    start_pos: int
    end_pos: int
    error_score: int

@dataclass
class Config:
    """
    Stores all configuration options for the evaluation process.
    This class is the single source of truth for default values.
    """
    short_text_threshold: int = 5
    long_text_threshold: int = 25 # New threshold for very long lines
    max_error_rate: float = 0.34
    max_error_rate_short: float = 0.32 # rate for short lines or text
    max_error_rate_long: float = 0.40 # More relaxed rate for very long lines
    discovery_error_rate: float = 0.40 # Permissive threshold for discovery phase
    plausibility_threshold: float = 0.70 # Jaccard index threshold for pre-filtering
    max_combinations: int = 120
    max_combination_gap: int = 10 # <-- NEW: Max gap between combined substrings
    min_combination_candidates: int = 10 # <-- NEW: Target for adaptive search
    case_conversion: bool = True
    unicode_normalization: bool = True
    width_conversion: bool = True
    punctuation_removal: bool = True
    whitespace_normalization: bool = True
    blank_line_removal: bool = True
    verbose: bool = False
    debug: bool = False  # Enable debug output for internal state tracing
    # Add original_file and ocr_file for argparse compatibility, though not used by the core logic directly.
    original_file: str = ""
    ocr_file: str = ""


# --- Core Logic Classes ---

class TextNormalizer:
    """
    Handles all text normalization steps based on the provided configuration.
    The normalization pipeline is applied in a specific, required order.
    """
    def __init__(self, config: Config):
        self.config = config

    def normalize(self, text: str) -> List[str]:
        """
        Applies the full normalization pipeline to a raw string.

        Args:
            text: The input string, potentially multi-line.

        Returns:
            A list of normalized text lines.
        """
        lines = text.split('\n')

        if self.config.case_conversion:
            lines = [line.lower() for line in lines]
        if self.config.unicode_normalization:
            lines = [unicodedata.normalize('NFC', line) for line in lines]
        if self.config.width_conversion:
            lines = [self._convert_full_width_to_half_width(line) for line in lines]
        if self.config.punctuation_removal:
            lines = [self._remove_punctuation(line) for line in lines]
        if self.config.whitespace_normalization:
            lines = [self._normalize_whitespace(line) for line in lines]
        if self.config.blank_line_removal:
            lines = [line for line in lines if line.strip()]

        return lines

    def _convert_full_width_to_half_width(self, text: str) -> str:
        """Converts full-width ASCII characters to their half-width equivalents."""
        return text.translate(str.maketrans(
            ''.join(chr(i) for i in range(0xFF01, 0xFF5F)),
            ''.join(chr(i - 0xFEE0) for i in range(0xFF01, 0xFF5F))
        ))

    def _remove_punctuation(self, text: str) -> str:
        """
        Removes punctuation, preserving dots/commas adjacent to digits.
        """
        result = []
        text_len = len(text)
        for i, char in enumerate(text):
            if char.isalnum() or char.isspace():
                result.append(char)
            elif char in '.,':
                is_adjacent_to_digit = (i > 0 and text[i - 1].isdigit()) or \
                                       (i < text_len - 1 and text[i + 1].isdigit())
                if is_adjacent_to_digit:
                    result.append(char)
        return ''.join(result)

    def _normalize_whitespace(self, text: str) -> str:
        """Replaces multiple whitespace characters with a single space and trims ends."""
        return re.sub(r'\s+', ' ', text).strip()


class SubstringMatcher:
    """
    Finds approximate substring matches using a memory and speed-optimized
    local alignment algorithm with pruning.
    """
    def __init__(self, config: Config):
        self.config = config

    def find_approximate_substrings(self, text: str, pattern: str) -> List[SubstringMatch]:
        """
        Finds all approximate matches of a pattern within a text using a
        modified Levenshtein DP algorithm (local alignment) with pruning.

        Args:
            text: The text to search within.
            pattern: The pattern to search for.

        Returns:
            A list of filtered, non-overlapping substring matches.
        """
        if not pattern or not text:
            return []

        text_len = len(text)
        pattern_len = len(pattern)
        max_errors = int(pattern_len * self.config.max_error_rate_short) \
            if pattern_len <= self.config.short_text_threshold else int(pattern_len * self.config.max_error_rate)

        prev_dp_row = [0] * (text_len + 1)
        prev_pos_row = list(range(text_len + 1))
        
        curr_dp_row = [0] * (text_len + 1)
        curr_pos_row = [0] * (text_len + 1)

        for i in range(1, pattern_len + 1):
            curr_dp_row[0] = i
            curr_pos_row[0] = 0
            min_in_row = i

            for j in range(1, text_len + 1):
                cost = 0 if pattern[i - 1] == text[j - 1] else 1
                
                sub_cost = prev_dp_row[j - 1] + cost
                del_cost = prev_dp_row[j] + 1
                ins_cost = curr_dp_row[j - 1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                curr_dp_row[j] = min_cost
                min_in_row = min(min_in_row, min_cost)

                if min_cost == sub_cost:
                    curr_pos_row[j] = prev_pos_row[j - 1]
                elif min_cost == ins_cost:
                    curr_pos_row[j] = curr_pos_row[j - 1]
                else:
                    curr_pos_row[j] = prev_pos_row[j]
            
            if min_in_row > max_errors:
                return [] # Pruning: if the best possible score in this row is too high, stop.

            prev_dp_row, curr_dp_row = curr_dp_row, prev_dp_row
            prev_pos_row, curr_pos_row = curr_pos_row, prev_pos_row

        final_dp_row = prev_dp_row
        final_pos_row = prev_pos_row
        
        matches = []
        for j in range(1, text_len + 1):
            error_score = final_dp_row[j]
            if error_score <= max_errors:
                start_pos = final_pos_row[j]
                if j > start_pos:
                    matches.append(SubstringMatch(-1, start_pos, j, error_score))

        return self._filter_overlapping_matches(matches)
        #return matches

    def _filter_overlapping_matches(self, matches: List[SubstringMatch]) -> List[SubstringMatch]:
        """Removes inferior overlapping matches, keeping the one with the lower error."""
        if not matches: return []
        matches.sort(key=lambda m: ( m.start_pos, m.error_score, -m.end_pos))
        filtered = []
        last_match = None
        for match in matches:
            if not last_match or match.start_pos >= last_match.end_pos:
                filtered.append(match)
                last_match = match
            elif match.error_score < last_match.error_score:
                filtered[-1] = match
                last_match = match
        return filtered


class MatchingEngine:
    """
    Implements the core two-phase matching algorithm to find the optimal
    set of matches between original and OCR lines.
    """
    def __init__(self, config: Config):
        self.config = config
        self.substring_matcher = SubstringMatcher(config)
        self.stats = {'candidates_found': 0, 'combinations_evaluated': 0, 'execution_time': 0}

    def find_optimal_matches(self, original_lines: List[str], ocr_lines: List[str]) -> Tuple[List[MatchCandidate], Dict]:
        """Finds the optimal line matches that minimize total error."""
        start_time = time.time()
        
        # Store lines temporarily for use in selection optimization
        self._current_original_lines = original_lines
        self._current_ocr_lines = ocr_lines
        
        candidates = self._generate_all_candidates(original_lines, ocr_lines)
        if self.config.verbose:
            print(f"Generated {len(candidates)} candidate matches.", file=sys.stderr)
        self.stats['candidates_found'] = len(candidates)
        optimal_matches = self._select_optimal_combination(candidates)
        self.stats['combinations_evaluated'] = len(optimal_matches)
        self.stats['execution_time'] = time.time() - start_time
        
        # Clean up temporary storage
        delattr(self, '_current_original_lines')
        delattr(self, '_current_ocr_lines')
        
        return optimal_matches, self.stats
    
    def _has_plausible_overlap(self, s1: str, s2: str) -> bool:
        """
        Quickly checks if two strings have enough common characters to be
        considered a potential match, using the Jaccard index. This acts as a
        fast pre-filter to avoid expensive calculations on dissimilar strings.
        """
        if not s1 or not s2:
            return False

        len_s1 = len(s1)
        len_s2 = len(s2)
        if min(len_s1, len_s2) < 3:
            return s1 in s2 if len_s1<=len_s2 else s2 in s1 # if two characters, must be substring (order considered)            
            
        set1 = set(s1)
        set2 = set(s2)
        
        intersection_len = len(set1.intersection(set2))
        if intersection_len == 0:
            return False
            
        min_len = min(len(set1), len(set2)) 

        jaccard_index = intersection_len / min_len
        if min_len < 6:
            return intersection_len >= min_len-1
        
        return jaccard_index >= self.config.plausibility_threshold

    def _generate_all_candidates(self, original_lines: List[str], ocr_lines: List[str]) -> List[MatchCandidate]:
        """
        Generates all types of candidate matches (1-to-1, N-to-1, 1-to-N).
        Enhanced to pre-calculate text lengths.
        """
        candidates = []
        
        # 1-to-1 matches
        for orig_idx, orig_line in enumerate(original_lines):
            if self.config.debug:
                print(f"(1-to-1)Processing Original line {orig_idx}: {orig_line}", file=sys.stderr)
            for ocr_idx, ocr_line in enumerate(ocr_lines):
                if not self._has_plausible_overlap(orig_line, ocr_line):
                    continue
                dist = levenshtein_distance(orig_line, ocr_line)
                if self._meets_error_threshold(orig_line, dist):
                    # Pre-calculate lengths
                    original_length = len(orig_line)
                    ocr_length = len(ocr_line)
                    
                    candidates.append(MatchCandidate(
                        [orig_idx], [ocr_idx], "1-to-1", dist,
                        original_length, ocr_length
                    ))

        # Debug output for 1-to-1 matches
        if self.config.debug:
            print(f"Found {len(candidates)} 1-to-1 matches.", file=sys.stderr)
            for match in candidates:
                print(f"  Match: {match.original_indices} -> {match.ocr_indices}, Error: {match.error_score}, Lengths: {match.original_text_length}, {match.ocr_text_length}", file=sys.stderr) 

        # N-to-1 matches
        for ocr_idx, ocr_line in enumerate(ocr_lines):
            substring_matches = []
            if self.config.debug:
                print(f"(N-to-1)Processing OCR line {ocr_idx}: {ocr_line}", file=sys.stderr)
            if self.config.debug and ocr_idx == 89:
                print(f"Debugging OCR line {ocr_idx}: {ocr_line}", file=sys.stderr) 
            
            for orig_idx, orig_line in enumerate(original_lines):
                #if self.config.debug:
                #    print(f"  Comparing with original line {orig_idx}: {orig_line}", file=sys.stderr)
                if self.config.debug and ocr_idx == 89 and orig_idx == 106:
                    print(f"Debugging match for OCR line {ocr_idx} and original line {orig_idx}.", file=sys.stderr)
                if not self._has_plausible_overlap(ocr_line, orig_line):
                    continue
                matches = self.substring_matcher.find_approximate_substrings(ocr_line, orig_line)
                for match in matches:
                    match.secondary_index = orig_idx
                    #if self.config.debug:
                    #    print(f"  Found match: {match.start_pos}-{match.end_pos} (Error: {match.error_score}) in original line [{orig_idx}] {orig_line}.", file=sys.stderr)
                    substring_matches.append(match)
            
            if substring_matches:
                #if self.config.debug and (ocr_idx == 8 and orig_idx == 11):
                #    print(f"Found {len(substring_matches)} substring matches for OCR line {ocr_idx}.", file=sys.stderr)
                
                # Pass original_lines for length calculation
                candidates.extend(self._process_multiline_combinations(
                    ocr_line, ocr_idx, substring_matches, original_lines, 'original'
                ))

        # 1-to-N matches  
        for orig_idx, orig_line in enumerate(original_lines):
            substring_matches = []
            if self.config.debug:
                print(f"(1-to-N)Processing Original line {orig_idx}: {orig_line}", file=sys.stderr)
            
            for ocr_idx, ocr_line in enumerate(ocr_lines):
                if self.config.debug and orig_idx == 174 and ocr_idx >= 0:
                    print(f"Debugging match for Original line {orig_idx} and OCR line {ocr_idx}.", file=sys.stderr)
                if not self._has_plausible_overlap(orig_line, ocr_line):
                    continue
                matches = self.substring_matcher.find_approximate_substrings(orig_line, ocr_line)
                for match in matches:
                    match.secondary_index = ocr_idx
                    substring_matches.append(match)

            if self.config.debug and orig_idx == 83:
                print(f"Debugging match for Original line {orig_idx}.", file=sys.stderr)            
            if substring_matches:
                # Pass ocr_lines for length calculation
                candidates.extend(self._process_multiline_combinations(
                    orig_line, orig_idx, substring_matches, ocr_lines, 'ocr'
                ))

        # print matched lines for debugging
        """
        if self.config.debug:
            print(f"Found {len(candidates)} 1-to-N matches.", file=sys.stderr)
            for match in candidates:
                if match.match_type != "1-to-N":
                    continue
                print(f"  Match: {match.original_indices} -> {match.ocr_indices}, Error: {match.error_score}", file=sys.stderr) 
                print(f"    Original[{match.original_indices[0]}]: {original_lines[match.original_indices[0]]}", file=sys.stderr)
                print(f"    OCR: {' '.join(ocr_lines[i]+f"({i})" for i in match.ocr_indices)} ", file=sys.stderr)
        """
        return candidates

    def _process_multiline_combinations(
        self, primary_text: str, primary_idx: int, substring_matches: List[SubstringMatch],
        secondary_lines: List[str], secondary_line_source: str
    ) -> List[MatchCandidate]:
        """
        Simplified approach that generates combinations with error rates during backtracking,
        sorts by quality, and processes only the best candidates.
        """
        if not substring_matches:
            return []
        
        # Step 1: Generate combinations with pre-calculated error rates
        combinations_with_rates = self._find_combinations_with_error_rates(
            substring_matches, primary_text, secondary_lines
        )
        
        if not combinations_with_rates:
            return []
        
        # Step 2: Sort by error rate (best combinations first)
        combinations_with_rates.sort(key=lambda x: (x[0], x[1], len(x[2])))  # error_rate, then combo size
        
        # Step 3: Process combinations until we have enough good candidates
        candidates = []
        perfect_match_found = False
        
        for error_rate, variance, combination in combinations_with_rates:
            # Early stopping if we have enough candidates and no perfect matches expected
            if (len(candidates) >= self.config.min_combination_candidates) or \
               ( perfect_match_found and len(candidates) >= self.config.min_combination_candidates/2):
                break
                
            # Evaluate this combination to get the actual MatchCandidate
            candidate = self._evaluate_combination_fast(
                combination, primary_text, primary_idx, secondary_lines, 
                secondary_line_source, estimated_error_rate=error_rate
            )
            
            if candidate:
                candidates.append(candidate)
                
                # Track if we found a perfect match
                if candidate.error_score == 0:
                    perfect_match_found = True
                    #break  # Perfect match found, no need to continue
        
        return candidates

    def _find_combinations_with_error_rates(
        self, matches: List[SubstringMatch], primary_text: str, 
        secondary_lines: List[str]
    ) -> List[Tuple[float, List[SubstringMatch]]]:
        """
        Find valid combinations and estimate their error rates during backtracking.
        Returns list of (estimated_error_rate, combination) tuples.
        """
        if not matches:
            return []
        
        # Sort matches for consistent processing
        matches = sorted(matches, key=lambda m: (m.start_pos, -m.end_pos, m.error_score, m.secondary_index))
        
        primary_length = len(primary_text)
        max_total_gap = self.config.max_combination_gap
        max_combos = self.config.max_combinations
        all_combinations = []
        
        def meets_basic_threshold(error_rate: float) -> bool:
            """Quick check if error rate is within reasonable bounds."""
            if primary_length <= self.config.short_text_threshold:
                return error_rate <= self.config.max_error_rate_short
            elif primary_length >= self.config.long_text_threshold:
                return error_rate <= self.config.max_error_rate_long
            else:
                return error_rate <= self.config.max_error_rate
        
        def backtrack(last_end: int, last_secondary_index: int, total_gap: int, accumulated_error: int, 
                    used_sec: Set[int], combo: List[SubstringMatch]):
            # If we have a valid multi-line combination, check if it meets threshold
            if len(combo) >= 2:
                # Simple unmatched calculation: primary text not covered up to last_end
                # This is an approximation - the actual calculation will be done in final evaluation
                unmatched_primary_length = primary_length - last_end if last_end > 0 else primary_length
                
                # Estimated total error = accumulated match errors + unmatched primary text penalty
                estimated_total_error = accumulated_error + unmatched_primary_length
                error_rate = estimated_total_error / primary_length if primary_length > 0 else float('inf')
                
                # Apply threshold based on primary text length
                meets_threshold = False
                if primary_length <= self.config.short_text_threshold:
                    meets_threshold = error_rate <= self.config.max_error_rate_short
                elif primary_length >= self.config.long_text_threshold:
                    meets_threshold = error_rate <= self.config.max_error_rate_long
                else:
                    meets_threshold = error_rate <= self.config.max_error_rate
                
                if meets_threshold:
                    # Calculate the standard deviation of positions of matched substrings
                    sum_pos = sum(m.secondary_index for m in combo)
                    mean_pos = sum_pos / len(combo)
                    variance = sum((m.secondary_index - mean_pos) ** 2 for m in combo) / len(combo)
                    all_combinations.append((error_rate, variance, combo.copy()))
            
            # Stop if it reached the end of primary text
            if last_end >= primary_length:
                return
            # Stop if we have enough combinations
            if len(all_combinations) >= max_combos:
                return
            
            # create rearranged matches to try to extend from the last_secondary_idx
            new_matches = []
            next_i = 0
            for i, match in enumerate(matches):
                if match.secondary_index > last_secondary_index:
                    new_matches.append(match)
                    if next_i == 0: next_i = i
            for i in range(0, next_i-1):
                new_matches.append(matches[i])

            # Try to extend the current combination
            for i, match in enumerate(new_matches):
                # Primary overlap check - ensure no overlap in primary text positions
                if last_end != -1 and match.start_pos < last_end:
                    continue
                
                # Start position must be after last_end but not too far away
                if (match.start_pos if last_end == -1 else (match.start_pos-last_end) ) > max_total_gap - total_gap:
                    continue
                    
                # Secondary uniqueness - ensure we don't use the same secondary line twice
                if match.secondary_index in used_sec:
                    continue

                # Calculate new total gap
                if last_end == -1:
                    gap = match.start_pos
                else:
                    gap = (match.start_pos-last_end-1) if primary_text[last_end] == ' ' and match.start_pos > last_end else (match.start_pos-last_end)
                new_total_gap = total_gap + gap

                # Calculate new accumulated error
                new_accumulated_error = accumulated_error + match.error_score + gap
                
                # Add this match and recurse
                combo.append(match)
                used_sec.add(match.secondary_index)
                backtrack(match.end_pos, match.secondary_index, new_total_gap, new_accumulated_error, used_sec, combo)
                used_sec.remove(match.secondary_index)
                combo.pop()
          
        # Start backtracking with initial accumulated values
        backtrack(-1, -1, 0, 0, set(), [])
        
        return all_combinations

    def _evaluate_combination_fast(
        self, combo: List[SubstringMatch], primary_text: str, primary_idx: int,
        secondary_lines: List[str], secondary_line_source: str, estimated_error_rate: float
    ) -> Optional[MatchCandidate]:
        """
        Fast evaluation that can skip some calculations since we have estimated error rate.
        """
        # Extract secondary indices
        sec_indices = [m.secondary_index for m in combo]
        
        # Test both concatenation methods (we know one will be close to estimated)
        concat_no_space = "".join(secondary_lines[i] for i in sec_indices)
        concat_with_space = " ".join(secondary_lines[i] for i in sec_indices)
        
        # Quick plausibility check
        if not self._has_plausible_overlap(primary_text, concat_no_space):
            return None
        
        # Calculate errors for both methods
        error_no_space = levenshtein_distance(primary_text, concat_no_space)
        error_with_space = levenshtein_distance(primary_text, concat_with_space)
        final_error = min(error_no_space, error_with_space)
        
        # Determine which concatenation method was used
        ref_text = concat_no_space if error_no_space <= error_with_space else concat_with_space
        
        # Final error threshold check (should pass since we pre-filtered, but double-check)
        if not self._meets_error_threshold(ref_text if secondary_line_source == 'original' else primary_text, final_error):
            return None
        
        # Pre-calculate text lengths
        primary_text_length = len(primary_text)
        secondary_text_length = len(ref_text)
        
        # Create and return candidate
        if secondary_line_source == 'original':
            return MatchCandidate(
                original_indices=sec_indices,
                ocr_indices=[primary_idx],
                match_type="N-to-1",
                error_score=final_error,
                original_text_length=secondary_text_length,
                ocr_text_length=primary_text_length
            )
        else:  # 'ocr'
            return MatchCandidate(
                original_indices=[primary_idx],
                ocr_indices=sec_indices,
                match_type="1-to-N",
                error_score=final_error,
                original_text_length=primary_text_length,
                ocr_text_length=secondary_text_length
            )
        
    def _is_perfect_candidate(self, candidate: MatchCandidate) -> bool:
        """
        Check if a candidate represents a perfect match with zero error.
        """
        return candidate.error_score == 0

    def _find_non_overlapping_combinations(
        self,
        matches: List[SubstringMatch],
        secondary_lines: List[str],
        current_gap: int,
        top_k: int = 3,
        score_fn: Optional[Callable[[SubstringMatch], float]] = None,
    ) -> List[List[SubstringMatch]]:
        """
        Return the TOP non-overlapping combinations ranked by a global objective,
        rather than just 'maximal' sets. This avoids greedy/local suboptimality
        (e.g., choosing one long span instead of two better-tight spans).

        Constraints enforced:
        - primary-axis non-overlap: next.start_pos >= last.end_pos
        - gap limit: (next.start_pos - last.end_pos) <= current_gap (if last exists)
        - uniqueness on secondary dimension: no duplicate secondary_index

        Scoring:
        By default, score(match) = length - α * error + β, where:
            - length = (end_pos - start_pos)
            - α = self.config.length_error_tradeoff (default 1.0)
            - β = self.config.bonus_per_match (default 0.25)
        You can pass a custom `score_fn` to fully control ranking.

        Returns:
        A list of combinations, best first. Each combination is a list[SubstringMatch].
        """
        if not matches:
            return []

        # Sort for stable exploration
        matches = sorted(matches, key=lambda m: (m.start_pos, m.end_pos, m.error_score, m.secondary_index))

        # Default scoring that can make {B,C} beat A when total errors tie,
        # via a small per-match bonus β.
        alpha = getattr(self.config, "length_error_tradeoff", 1.0)
        beta  = getattr(self.config, "bonus_per_match", 0.25)

        def _default_score(m: SubstringMatch) -> float:
            length = max(0, m.end_pos - m.start_pos)
            return float(length) - alpha * float(m.error_score) + beta

        score_match = score_fn or _default_score

        max_depth = getattr(self.config, "max_combination_lines", None)

        # Maintain a small top-K of best combos
        import heapq
        best_heap: List[Tuple[float, Tuple[int, ...], List[SubstringMatch]]] = []  # (score, key, combo)
        seen_keys: Set[Tuple[int, ...]] = set()

        def push_combo(combo: List[SubstringMatch]) -> None:
            # Keyed by the set of secondary indices (order-insensitive) to dedupe
            key = tuple(sorted(m.secondary_index for m in combo))
            if key in seen_keys:
                return
            seen_keys.add(key)
            total = sum(score_match(m) for m in combo)
            heapq.heappush(best_heap, (total, key, combo.copy()))
            if len(best_heap) > top_k:
                heapq.heappop(best_heap)

        # Backtracking with constraints; choose globally best by score
        def backtrack(start_idx: int, last_end: int, used_sec: Set[int], combo: List[SubstringMatch]):
            if max_depth is not None and len(combo) >= max_depth:
                if len(combo) >= 1:
                    push_combo(combo)
                return

            extended = False
            for i in range(start_idx, len(matches)):
                m = matches[i]

                # Primary overlap check
                if last_end != -1 and m.start_pos < last_end:
                    continue
                # Gap constraint
                if last_end != -1 and (m.start_pos - last_end) > current_gap:
                    continue
                # Secondary uniqueness
                if m.secondary_index in used_sec:
                    continue

                extended = True
                combo.append(m)
                used_sec.add(m.secondary_index)
                backtrack(i + 1, m.end_pos, used_sec, combo)
                used_sec.remove(m.secondary_index)
                combo.pop()

            # If we couldn't extend, finalize the current combo
            if not extended and len(combo) >= 1:
                push_combo(combo)

        backtrack(0, -1, set(), [])

        # Return best-first
        best = sorted(best_heap, key=lambda t: t[0], reverse=True)
        return [combo for _, __, combo in best]

    def _meets_error_threshold(self, original_text: str, error_score: int) -> bool:
        """
        Checks if an error score is within the acceptable threshold using a
        three-tiered logic based on text length.
        """
        if not original_text: return False
        text_len = len(original_text)
        
        if text_len < self.config.short_text_threshold:
            return error_score <= 1
            
        if text_len >= self.config.long_text_threshold:
            return (error_score / text_len) <= self.config.max_error_rate_long
            
        return (error_score / text_len) <= self.config.max_error_rate

    def _select_optimal_combination(self, candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        """
        DEBUG VERSION: Selects the best set of non-conflicting matches with extensive debugging.
        """
        if not candidates: 
            return []
        
        original_lines = getattr(self, '_current_original_lines', [])
        ocr_lines = getattr(self, '_current_ocr_lines', [])
        
        # DEBUG: Look for specific candidates involving L106, L107, L108, L089
        print(f"\n=== DEBUGGING CANDIDATE SELECTION ===", file=sys.stderr)
        print(f"Total candidates: {len(candidates)}", file=sys.stderr)
        
        # Look for candidates involving the problematic lines
        target_orig_indices = {106, 107, 108}  # Adjust these to match your actual line numbers
        target_ocr_index = 89                  # Adjust this to match your actual OCR line number
        
        relevant_candidates = []
        for i, candidate in enumerate(candidates):
            orig_overlap = set(candidate.original_indices) & target_orig_indices
            ocr_overlap = target_ocr_index in candidate.ocr_indices
            
            if orig_overlap or ocr_overlap:
                relevant_candidates.append((i, candidate))

                """
                print(f"Candidate {i}: {candidate.original_indices} -> {candidate.ocr_indices}", file=sys.stderr)
                print(f"  Type: {candidate.match_type}, Error: {candidate.error_score}", file=sys.stderr)
                print(f"  Text lengths: orig={candidate.original_text_length}, ocr={candidate.ocr_text_length}", file=sys.stderr)
                
                # Show actual text content
                orig_text = " ".join(original_lines[j] for j in candidate.original_indices)
                ocr_text = " ".join(ocr_lines[j] for j in candidate.ocr_indices)
                print(f"  Original: '{orig_text}'", file=sys.stderr)
                print(f"  OCR: '{ocr_text}'", file=sys.stderr)
                """
                # Calculate error savings
                unmatched_orig_penalty = sum(len(original_lines[j]) for j in candidate.original_indices)
                unmatched_ocr_penalty = sum(len(ocr_lines[j]) for j in candidate.ocr_indices)
                savings = (unmatched_orig_penalty + unmatched_ocr_penalty) - candidate.error_score
                print(f"  Error savings: {savings} (penalty: {unmatched_orig_penalty + unmatched_ocr_penalty}, error: {candidate.error_score})", file=sys.stderr)
                print(f"", file=sys.stderr)
        
        print(f"Found {len(relevant_candidates)} relevant candidates", file=sys.stderr)
        
        # Step 1: Get reference solution using original greedy approach
        reference_matches = self._original_greedy_selection(candidates)
        reference_total_error = self._calculate_total_error_including_unmatched(
            reference_matches, original_lines, ocr_lines)
        
        print(f"\n=== REFERENCE SOLUTION ===", file=sys.stderr)
        print(f"Reference matches: {len(reference_matches)}", file=sys.stderr)
        reference_match_error = sum(match.error_score for match in reference_matches)
        print(f"Match error: {reference_match_error}, Total error: {reference_total_error}", file=sys.stderr)
        
        # Show which of our target lines got matched in reference solution
        ref_used_orig = set()
        ref_used_ocr = set()
        for match in reference_matches:
            ref_used_orig.update(match.original_indices)
            ref_used_ocr.update(match.ocr_indices)
            
            # Check if this match involves our target lines
            orig_overlap = set(match.original_indices) & target_orig_indices
            ocr_overlap = target_ocr_index in match.ocr_indices
            if orig_overlap or ocr_overlap:
                print(f"Reference used target lines: {match.original_indices} -> {match.ocr_indices} (error: {match.error_score})", file=sys.stderr)
        
        print(f"Target original lines status:", file=sys.stderr)
        for idx in target_orig_indices:
            if idx < len(original_lines):
                status = "USED" if idx in ref_used_orig else "UNUSED"
                print(f"  L{idx}: {status} - '{original_lines[idx]}'", file=sys.stderr)
        
        if target_ocr_index < len(ocr_lines):
            status = "USED" if target_ocr_index in ref_used_ocr else "UNUSED"
            print(f"Target OCR line L{target_ocr_index}: {status} - '{ocr_lines[target_ocr_index]}'", file=sys.stderr)
        
        # Step 2: Try to find better solution using bounded backtracking
        if len(candidates) <= 400:
            print(f"\n=== STARTING BACKTRACKING ===", file=sys.stderr)
            better_matches = self._bounded_backtracking_search(
                candidates, reference_total_error, original_lines, ocr_lines)
            if better_matches is not None:
                better_total_error = self._calculate_total_error_including_unmatched(
                    better_matches, original_lines, ocr_lines)
                better_match_error = sum(match.error_score for match in better_matches)
                print(f"Improved solution found: {len(better_matches)} matches", file=sys.stderr)
                print(f"Match error: {better_match_error}, Total error: {better_total_error}", file=sys.stderr)
                return better_matches
            else:
                print(f"No improved solution found", file=sys.stderr)
        
        print(f"Using reference solution", file=sys.stderr)
        return reference_matches

    # 4. ADD these new methods to the MatchingEngine class:

    def _original_greedy_selection(self, candidates: List[MatchCandidate]) -> List[MatchCandidate]:
        """Original greedy selection algorithm to establish baseline."""
        def sort_key(candidate: MatchCandidate) -> tuple:
            error_rate = candidate.error_score / max(candidate.original_text_length, 1)
            return (
                error_rate, 
                -candidate.original_text_length,
                len(candidate.original_indices) + len(candidate.ocr_indices)
            )
        
        sorted_candidates = sorted(candidates, key=sort_key)
        
        selected_matches = []
        used_original_indices = set()
        used_ocr_indices = set()
        
        for candidate in sorted_candidates:
            is_orig_conflict = any(i in used_original_indices for i in candidate.original_indices)
            is_ocr_conflict = any(i in used_ocr_indices for i in candidate.ocr_indices)
            
            if not is_orig_conflict and not is_ocr_conflict:
                selected_matches.append(candidate)
                used_original_indices.update(candidate.original_indices)
                used_ocr_indices.update(candidate.ocr_indices)
        
        return selected_matches

    def _bounded_backtracking_search(self, candidates: List[MatchCandidate], 
                                    reference_total_error: int,
                                    original_lines: List[str], 
                                    ocr_lines: List[str]) -> Optional[List[MatchCandidate]]:
        """
        Use backtracking to find a better solution than the reference, with aggressive pruning
        based on TOTAL error including unmatched line penalties.
        
        Returns None if no better solution is found, otherwise returns the improved match list.
        """
        # Calculate "error savings" for each candidate - how much total error it saves
        # compared to leaving those lines unmatched
        def calculate_error_savings(candidate: MatchCandidate) -> float:
            """Calculate how much total error this candidate saves vs leaving lines unmatched."""
            # Penalty if these lines were left unmatched
            unmatched_orig_penalty = sum(len(original_lines[i]) for i in candidate.original_indices)
            unmatched_ocr_penalty = sum(len(ocr_lines[i]) for i in candidate.ocr_indices)
            unmatched_total_penalty = unmatched_orig_penalty + unmatched_ocr_penalty
            
            # Error savings = penalty avoided - match error
            savings = unmatched_total_penalty - candidate.error_score
            
            # Add bonus for multi-line matches (they often represent important layout reconstruction)
            line_count = len(candidate.original_indices) + len(candidate.ocr_indices)
            multiline_bonus = 2.0 if line_count > 2 else 0.0
            
            return savings + multiline_bonus
        
        # Sort candidates by error savings (highest savings first) 
        # This ensures we try the most valuable candidates first
        candidates_with_savings = [(calculate_error_savings(c), c) for c in candidates]
        candidates_with_savings.sort(key=lambda x: (-x[0], x[1].error_score))  # Best savings first, then lowest error
        candidates = [c for _, c in candidates_with_savings]
        
        if self.config.debug:
            print(f"Top 5 candidates by error savings:", file=sys.stderr)
            for i, (savings, candidate) in enumerate(candidates_with_savings[:5]):
                print(f"  {i+1}. Savings: {savings:.1f}, Error: {candidate.error_score}, "
                    f"Match: {candidate.original_indices} -> {candidate.ocr_indices} ({candidate.match_type})", file=sys.stderr)
        
        best_matches = None
        best_total_error = reference_total_error
        nodes_explored = 0
        max_nodes = 15000  # Increased limit since we have better candidate ordering
        
        def calculate_current_total_error(current_matches: List[MatchCandidate]) -> int:
            """Calculate total error for current partial solution including unmatched penalties."""
            # Error from current matches
            matched_error = sum(match.error_score for match in current_matches)
            
            # Get used line indices from current matches
            used_orig_indices = set()
            used_ocr_indices = set()
            for match in current_matches:
                used_orig_indices.update(match.original_indices)
                used_ocr_indices.update(match.ocr_indices)
            
            # Penalties for currently unmatched lines
            unmatched_orig_penalty = sum(len(line) for i, line in enumerate(original_lines) 
                                        if i not in used_orig_indices)
            unmatched_ocr_penalty = sum(len(line) for i, line in enumerate(ocr_lines) 
                                    if i not in used_ocr_indices)
            
            return matched_error + unmatched_orig_penalty + unmatched_ocr_penalty
        
        def backtrack(idx: int, current_matches: List[MatchCandidate], 
                    used_orig: set, used_ocr: set) -> bool:
            nonlocal best_matches, best_total_error, nodes_explored
            
            nodes_explored += 1
            if nodes_explored > max_nodes:
                return True  # Stop exploration due to computational limit
            
            # Calculate current total error including unmatched penalties
            current_total_error = calculate_current_total_error(current_matches)
            
            # Aggressive pruning: if current total error already meets or exceeds reference, stop
            if current_total_error >= best_total_error:
                return False
            
            # Base case: evaluated all candidates
            if idx == len(candidates):
                if current_total_error < best_total_error:
                    best_total_error = current_total_error
                    best_matches = current_matches.copy()
                    if self.config.debug:
                        match_error = sum(match.error_score for match in current_matches)
                        print(f"  New best: {len(current_matches)} matches, match_error: {match_error}, total_error: {current_total_error}", file=sys.stderr)
                return False
            
            candidate = candidates[idx]
            
            # Check if this candidate conflicts with current selection
            orig_conflict = any(i in used_orig for i in candidate.original_indices)
            ocr_conflict = any(i in used_ocr for i in candidate.ocr_indices)
            
            # Try including this candidate (if no conflict)
            if not orig_conflict and not ocr_conflict:
                current_matches.append(candidate)
                new_used_orig = used_orig | set(candidate.original_indices)
                new_used_ocr = used_ocr | set(candidate.ocr_indices)
                
                if backtrack(idx + 1, current_matches, new_used_orig, new_used_ocr):
                    return True  # Early termination due to computational limit
                
                current_matches.pop()
            
            # Try skipping this candidate
            return backtrack(idx + 1, current_matches, used_orig, used_ocr)
        
        if self.config.debug:
            print(f"Starting bounded backtracking search with reference total error: {reference_total_error}", file=sys.stderr)
        
        backtrack(0, [], set(), set())
        
        if self.config.debug:
            print(f"Backtracking explored {nodes_explored} nodes", file=sys.stderr)
        
        return best_matches  # None if no improvement found

    def _calculate_total_error_including_unmatched(self, matches: List[MatchCandidate], 
                                                original_lines: List[str], 
                                                ocr_lines: List[str]) -> int:
        """
        Calculate the true total error score including penalties for unmatched lines.
        This is what should be minimized for the overall OCR evaluation.
        """
        # Error from matched lines
        matched_error = sum(match.error_score for match in matches)
        
        # Get used line indices
        used_orig_indices = set()
        used_ocr_indices = set()
        for match in matches:
            used_orig_indices.update(match.original_indices)
            used_ocr_indices.update(match.ocr_indices)
        
        # Penalties for unmatched lines (full character count)
        unmatched_orig_penalty = sum(len(line) for i, line in enumerate(original_lines) 
                                    if i not in used_orig_indices)
        unmatched_ocr_penalty = sum(len(line) for i, line in enumerate(ocr_lines) 
                                if i not in used_ocr_indices)
        
        return matched_error + unmatched_orig_penalty + unmatched_ocr_penalty



class OCREvaluator:
    """The main class that orchestrates the entire OCR evaluation process."""
    def __init__(self, config: Config):
        self.config = config
        self.normalizer = TextNormalizer(config)
        self.matching_engine = MatchingEngine(config)

    def evaluate(self, original_text: str, ocr_text: str) -> Dict:
        """Performs the full evaluation and returns a dictionary of results."""
        original_lines = self.normalizer.normalize(original_text)
        ocr_lines = self.normalizer.normalize(ocr_text)
        matches, stats = self.matching_engine.find_optimal_matches(original_lines, ocr_lines)
        
        # --- Calculate all metrics based on the final matches ---
        
        # Get sets of used line indices
        used_originals = set()
        used_ocr = set()
        for match in matches:
            used_originals.update(match.original_indices)
            used_ocr.update(match.ocr_indices)

        # Calculate error scores
        matched_error_score = sum(match.error_score for match in matches)
        unmatched_original_error = sum(len(line) for i, line in enumerate(original_lines) if i not in used_originals)
        unmatched_ocr_error = sum(len(line) for i, line in enumerate(ocr_lines) if i not in used_ocr)
        total_error_score = matched_error_score + unmatched_original_error + unmatched_ocr_error

        # Calculate character counts
        matched_original_chars_count = sum(len(line) for i, line in enumerate(original_lines) if i in used_originals)
        
        # Calculate line counts
        matched_original_lines_count = len(used_originals)
        unmatched_original_lines_count = len(original_lines) - len(used_originals)
        unmatched_ocr_lines_count = len(ocr_lines) - len(used_ocr)

        return {
            'total_error_score': total_error_score,
            'matched_error_score': matched_error_score,
            'unmatched_original_error': unmatched_original_error,
            'unmatched_ocr_error': unmatched_ocr_error,
            
            'matched_original_chars_count': matched_original_chars_count,
            
            'matched_original_lines_count': matched_original_lines_count,
            'unmatched_original_lines_count': unmatched_original_lines_count,
            'unmatched_ocr_lines_count': unmatched_ocr_lines_count,
            
            'original_line_count': len(original_lines),
            'ocr_line_count': len(ocr_lines),
            'original_char_count': sum(len(line) for line in original_lines),
            'ocr_char_count': sum(len(line) for line in ocr_lines),
            
            'matches': matches,
            'used_original_indices': used_originals,
            'used_ocr_indices': used_ocr,
            'original_lines': original_lines,
            'ocr_lines': ocr_lines,
            'stats': stats,
        }

# --- Utility Functions ---

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein distance using a space-optimized algorithm."""
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if not s2: return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def format_output(results: Dict, config: Config) -> str:
    """Formats the evaluation results into a human-readable string based on the new spec."""
    output = [
        f"===== OCR Evaluation Results (v{__version__}) =====",
        f"FILES: Original File: {config.original_file}, OCR File: {config.ocr_file}",
        f"TOTAL ERROR SCORE: {results['total_error_score']}",
        f"ERROR SCORE BREAK DOWN : {results['matched_error_score']}, {results['unmatched_original_error']}, {results['unmatched_ocr_error']}",
        f"NO. OF CHARS : {results['matched_original_chars_count']}, {results['unmatched_original_error']}, {results['unmatched_ocr_error']}",
        f"NO. OF LINES : {results['matched_original_lines_count']}, {results['unmatched_original_lines_count']}, {results['unmatched_ocr_lines_count']}",
        f"NO. OF LINES IN FILE: {results['original_line_count']}, {results['ocr_line_count']}",
        f"NO. OF CHARS IN FILE: {results['original_char_count']}, {results['ocr_char_count']}",
    ]

    if config.verbose:
        output.append("\n" + "="*20 + " VERBOSE OUTPUT " + "="*20)
        sorted_matches = sorted(results['matches'], key=lambda m: m.original_indices[0])
        
        output.append("\n--- Matched Line Pairs ---")
        if not sorted_matches: output.append("  None")
        
        for i, match in enumerate(sorted_matches):
            output.append(f"\nMatch {i+1}: Type={match.match_type}, Error={match.error_score}")
            output.append(f"  Original (Indices: {match.original_indices})")
            for j in match.original_indices: output.append(f"    L{j:03d}: {results['original_lines'][j]}")
            if len(match.original_indices) > 1: output.append(f"    => Combined: \"{' '.join(results['original_lines'][j] for j in match.original_indices)}\"")
            output.append(f"  OCR (Indices: {match.ocr_indices})")
            for j in match.ocr_indices: output.append(f"    L{j:03d}: {results['ocr_lines'][j]}")
            if len(match.ocr_indices) > 1: output.append(f"    => Combined: \"{' '.join(results['ocr_lines'][j] for j in match.ocr_indices)}\"")
        
        output.append("\n--- Unmatched Original Lines (Deletions) ---")
        unmatched_orig = [f"  L{i:03d}: {line} (Penalty: {len(line)})" for i, line in enumerate(results['original_lines']) if i not in results['used_original_indices']]
        output.extend(unmatched_orig if unmatched_orig else ["  None"])

        output.append("\n--- Unmatched OCR Lines (Insertions) ---")
        unmatched_ocr = [f"  L{i:03d}: {line} (Penalty: {len(line)})" for i, line in enumerate(results['ocr_lines']) if i not in results['used_ocr_indices']]
        output.extend(unmatched_ocr if unmatched_ocr else ["  None"])

        stats = results['stats']
        output.append("\n--- Algorithm Statistics ---")
        output.append(f"  Execution time: {stats['execution_time']:.3f} seconds")
        output.append(f"  Candidate matches found: {stats['candidates_found']}")
        output.append(f"  Optimal matches selected: {len(results['matches'])}")
    return "\n".join(output)

def load_file(filepath: str) -> str:
    """Loads a text file with robust error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr); sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: File '{filepath}' is not valid UTF-8.", file=sys.stderr); sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when reading '{filepath}'.", file=sys.stderr); sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr); sys.exit(1)

# --- Main Execution ---

def main():
    """Parses command-line arguments and runs the evaluation."""
    defaults = Config()

    parser = argparse.ArgumentParser(
        description="A robust OCR quality evaluation tool.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example: python %(prog)s original.txt ocr_output.txt -v"
    )
    parser.add_argument("original_file", help="Path to the original text file (ground truth).")
    parser.add_argument("ocr_file", help="Path to the OCR output text file.")
    
    tuning_group = parser.add_argument_group('Algorithm Tuning')
    tuning_group.add_argument("--short-text-threshold", type=int, default=defaults.short_text_threshold, 
                              help=f"Line length to be considered 'short'. Default: {defaults.short_text_threshold}.")
    tuning_group.add_argument("--long-text-threshold", type=int, default=defaults.long_text_threshold,
                              help=f"Line length to be considered 'very long' for a more relaxed error rate. Default: {defaults.long_text_threshold}.")
    tuning_group.add_argument("--max-error-rate", type=float, default=defaults.max_error_rate, 
                              help=f"Max error rate for matching normal length lines. Default: {defaults.max_error_rate}.")
    tuning_group.add_argument("--max-error-rate-long", type=float, default=defaults.max_error_rate_long,
                              help=f"Max error rate for matching very long lines. Default: {defaults.max_error_rate_long}.")
    tuning_group.add_argument("--discovery-error-rate", type=float, default=defaults.discovery_error_rate, 
                              help=f"Permissive error rate for substring discovery. Default: {defaults.discovery_error_rate}.")
    tuning_group.add_argument("--plausibility-threshold", type=float, default=defaults.plausibility_threshold, 
                              help=f"Jaccard index threshold for pre-filtering. Default: {defaults.plausibility_threshold}.")
    tuning_group.add_argument("--max-combinations", type=int, default=defaults.max_combinations, 
                              help=f"Max lines in an N-to-1 or 1-to-N match. Default: {defaults.max_combinations}.")
    # <-- NEW Command-line argument -->
    tuning_group.add_argument("--max-combination-gap", type=int, default=defaults.max_combination_gap, 
                              help=f"Max character gap between combined substrings. Default: {defaults.max_combination_gap}.")

    norm_group = parser.add_argument_group('Normalization Control')
    norm_group.add_argument("--disable-case-conversion", action="store_false", dest="case_conversion", help="Disable converting text to lowercase.")
    norm_group.add_argument("--disable-unicode-normalization", action="store_false", dest="unicode_normalization", help="Disable Unicode NFC normalization.")
    norm_group.add_argument("--disable-width-conversion", action="store_false", dest="width_conversion", help="Disable converting full-width to half-width characters.")
    norm_group.add_argument("--disable-punctuation-removal", action="store_false", dest="punctuation_removal", help="Disable removal of most punctuation.")
    norm_group.add_argument("--disable-whitespace-normalization", action="store_false", dest="whitespace_normalization", help="Disable normalizing multiple spaces to one and trimming.")
    norm_group.add_argument("--disable-blank-line-removal", action="store_false", dest="blank_line_removal", help="Disable removing blank lines.")

    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument("-v", "--verbose", action="store_true", help="Enable detailed verbose output.")
    output_group.add_argument("--debug", action="store_true", help="Enable debug output for internal state tracing.")
    output_group.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()

    config = Config(**vars(args))
    
    original_text = load_file(config.original_file)
    ocr_text = load_file(config.ocr_file)
    
    evaluator = OCREvaluator(config)
    results = evaluator.evaluate(original_text, ocr_text)
    
    print(format_output(results, config))

if __name__ == "__main__":
    main()

def evaluate_files(original_file, ocr_file, verbose=False):
    """
    Programmatic entrypoint: compares two text files and returns the formatted evaluation string.
    """
    config = Config(original_file=original_file, ocr_file=ocr_file, verbose=verbose)
    original_text = load_file(config.original_file)
    ocr_text = load_file(config.ocr_file)

    evaluator = OCREvaluator(config)
    results = evaluator.evaluate(original_text, ocr_text)
    return format_output(results, config)
