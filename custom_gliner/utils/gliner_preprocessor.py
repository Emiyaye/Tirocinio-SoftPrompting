import os
from collections import Counter
from datasets import load_dataset, load_from_disk
import ast
import re

SEED = 42

class GlinerDataPreprocessor:
    """
    A preprocessor for NER datasets compatible with GLiNER format.
    Automatically handles train/validation/test splits and converts data to spans format.
    """    
    def __init__(self,
                 dataset_name: str,
                 val_split_ratio: float = 0.1,
                 dataset_subset: float = 1.0,
                 convert_to_spans: bool = True,
                 filter_empty_entities: bool = True,
                 text_column_name: str = "tokens",
                 data_dir: str = None,
                 max_input_length: int = None,):
        self.dataset_name = dataset_name
        self.val_split_ratio = val_split_ratio
        self.dataset_subset = dataset_subset
        self.filter_empty_entities = filter_empty_entities
        self.text_column_name = text_column_name
        self.data_dir = data_dir
        self.max_input_length = max_input_length

        # Validate parameters
        if not 0.0 < dataset_subset <= 1.0:
            raise ValueError("dataset_subset must be between 0.0 (exclusive) and 1.0 (inclusive)")
        if not 0.0 <= val_split_ratio <= 0.5:
            raise ValueError("val_split_ratio must be between 0.0 and 0.5")

        self._load_dataset()
        self._apply_dataset_subset()
        self._prepare_splits()
        if self.max_input_length is not None:
            self._filter_by_length()
        if convert_to_spans:
            self._convert_to_spans()
            self.ner_tags = self._get_ner_tags()
        else:
            self._prepare_raw_format()
            self.ner_tags = self._get_ner_tags_raw()
            self.ner_tags_bio = self._get_ner_tags_bio_raw()

    def validate_val_split_ratio(cls, v):
        if not 0.0 <= v <= 0.5:
            raise ValueError("val_split_ratio must be between 0.0 and 0.5")
        return v

    def _name_to_slug(self, name: str) -> str:
        return name.replace("/", "--")

    def _load_dataset(self):
        """Load the dataset from HuggingFace or from local disk."""
        if self.data_dir is not None:
            local_path = os.path.join(self.data_dir, self._name_to_slug(self.dataset_name))
            print(f"  Loading dataset from disk: {local_path}")
            self.ds = load_from_disk(local_path)
        else:
            self.ds = load_dataset(self.dataset_name)
        self._parse_stringified_columns()

    def _parse_stringified_columns(self):
        """Auto-detect and parse columns stored as string representations of lists.
        
        Some HuggingFace datasets (e.g. disi-unibo-nlp/Pile-NER-biomed-IOB) store
        'tokens' and 'ner_tags' as serialized Python list strings instead of actual
        list/sequence types. This method detects such columns and converts them.
        """
        for split_name in self.ds.keys():
            features = self.ds[split_name].features
            columns_to_parse = []
            for col_name, feature in features.items():
                # Check if the column is a plain string Value (not Sequence)
                if hasattr(feature, 'dtype') and feature.dtype == 'string':
                    # Peek at first row to see if it looks like a stringified list
                    sample_val = self.ds[split_name][0][col_name]
                    if isinstance(sample_val, str) and sample_val.strip().startswith('['):
                        columns_to_parse.append(col_name)
            
            if columns_to_parse:
                print(f"  Parsing stringified list columns in '{split_name}': {columns_to_parse}")
                for col in columns_to_parse:
                    self.ds[split_name] = self.ds[split_name].map(
                        lambda x, c=col: {c: ast.literal_eval(x[c])},
                    )

    def _apply_dataset_subset(self):
        """Apply dataset subsetting if dataset_subset < 1.0."""
        if self.dataset_subset < 1.0:
            print(f"\nApplying dataset subsetting ({self.dataset_subset * 100:.1f}% of original data):")
            for split_name in self.ds.keys():
                split_size = len(self.ds[split_name])
                subset_size = int(split_size * self.dataset_subset)
                if subset_size > 0:
                    # Use deterministic sampling (first N samples) for reproducibility
                    indices = range(subset_size)
                    self.ds[split_name] = self.ds[split_name].select(indices)
                    print(f"  {split_name}: {split_size} -> {subset_size} samples")
                else:
                    print(f"  Warning: {split_name} subset size is 0, keeping at least 1 sample")
                    self.ds[split_name] = self.ds[split_name].select([0])

    def _prepare_splits(self):
        """Prepare train/val/test splits based on available data."""
        available_splits = list(self.ds.keys())
        
        if 'train' in available_splits and 'validation' in available_splits and 'test' in available_splits:
            # All splits available
            self._raw_train = self.ds['train']
            self._raw_val = self.ds['validation']
            self._raw_test = self.ds['test']
        
        elif 'train' in available_splits and 'test' in available_splits:
            # No validation set - create one from train
            train_val_split = self.ds['train'].train_test_split(
                test_size=self.val_split_ratio, 
                seed=SEED
            )
            self._raw_train = train_val_split['train']
            self._raw_val = train_val_split['test']
            self._raw_test = self.ds['test']
        
        elif 'train' in available_splits:
            # Only train available - split into train/val/test
            train_rest = self.ds['train'].train_test_split(
                test_size=self.val_split_ratio * 2,
                seed=SEED
            )
            val_test = train_rest['test'].train_test_split(
                test_size=0.5,
                seed=SEED
            )
            self._raw_train = train_rest['train']
            self._raw_val = val_test['train']
            self._raw_test = val_test['test']
        
        else:
            raise ValueError(f"Dataset must have at least 'train' split. Available: {available_splits}")
    
    def _filter_by_length(self):
        """Remove entries where the number of words exceeds max_input_length.

        Operates on the raw HuggingFace Dataset splits before span conversion.
        This prevents sequences from exceeding BERT's 512 position limit after
        soft prompt tokens are prepended.
        """
        col = self.text_column_name
        max_len = self.max_input_length
        print(f"\nFiltering entries with more than {max_len} words:")
        for split_name, attr in [("train", "_raw_train"), ("val", "_raw_val"), ("test", "_raw_test")]:
            ds = getattr(self, attr)
            orig_size = len(ds)
            ds = ds.filter(lambda x: len(x[col]) <= max_len)
            setattr(self, attr, ds)
            print(f"  {split_name}: {orig_size} -> {len(ds)} ({orig_size - len(ds)} removed)")

    def _filter_empty_entities(self, samples_list):
        """Remove entries that have no entity spans (only O tags).

        Args:
            samples_list: List of samples with 'ner' key containing entity spans.

        Returns:
            Filtered list containing only samples with at least one entity.
        """
        return [sample for sample in samples_list if len(sample['ner']) > 0]

    def _filter_empty_entities_raw(self, samples_list):
        """Remove entries where all NER tags are 'O' (no entities).

        Args:
            samples_list: List of samples with 'ner_tags' key containing BIO tags.

        Returns:
            Filtered list containing only samples with at least one non-O tag.
        """
        return [sample for sample in samples_list if any(tag != 'O' for tag in sample['ner_tags'])]

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalize an entity type label: replace underscores with spaces and lowercase."""
        return label.replace('_', ' ').lower()

    @staticmethod
    def _extract_tags_present(ner_tags):
        """Extract sorted unique entity types from BIO tags, stripping B-/I- prefixes."""
        return sorted({
            GlinerDataPreprocessor._normalize_label(tag[2:])
            for tag in ner_tags
            if tag != 'O' and (tag.startswith('B-') or tag.startswith('I-'))
        })
    
    @staticmethod
    def _extract_tags_present_bio(ner_tags):
        """Extract sorted unique entity types from BIO tags, without stripping B-/I- prefixes."""
        return sorted({
            GlinerDataPreprocessor._normalize_label(tag)
            for tag in ner_tags
            if tag != 'O' and (tag.startswith('B-') or tag.startswith('I-'))
        })

    def _prepare_raw_format(self):
        """Keep data in raw token/BIO-tag format and filter empty entities."""
        self.ds_train = [{'tokens': s[self.text_column_name], 'ner_tags': s['ner_tags'], 'tags_present': self._extract_tags_present(s['ner_tags']), 'tag_present_bio': self._extract_tags_present_bio(s['ner_tags'])} for s in self._raw_train]
        self.ds_val = [{'tokens': s[self.text_column_name], 'ner_tags': s['ner_tags'], 'tags_present': self._extract_tags_present(s['ner_tags']), 'tag_present_bio': self._extract_tags_present_bio(s['ner_tags'])} for s in self._raw_val]
        self.ds_test = [{'tokens': s[self.text_column_name], 'ner_tags': s['ner_tags'], 'tags_present': self._extract_tags_present(s['ner_tags']), 'tag_present_bio': self._extract_tags_present_bio(s['ner_tags'])} for s in self._raw_test]

        orig_train = len(self.ds_train)
        orig_val = len(self.ds_val)
        orig_test = len(self.ds_test)

        if self.filter_empty_entities:
            self.ds_train = self._filter_empty_entities_raw(self.ds_train)
            self.ds_val = self._filter_empty_entities_raw(self.ds_val)
            self.ds_test = self._filter_empty_entities_raw(self.ds_test)

        print(f"Filtered entries with no entities (raw format):")
        print(f"  Train: {orig_train} -> {len(self.ds_train)} ({orig_train - len(self.ds_train)} removed)")
        print(f"  Val: {orig_val} -> {len(self.ds_val)} ({orig_val - len(self.ds_val)} removed)")
        print(f"  Test: {orig_test} -> {len(self.ds_test)} ({orig_test - len(self.ds_test)} removed)")
        self._print_entity_distribution(mode="raw")

    def _convert_to_spans(self):
        """Convert all splits to GLiNER span format and filter empty entities."""
        self.ds_train = [self.ner_tags_to_spans(i) for i in self._raw_train]
        self.ds_val = [self.ner_tags_to_spans(i) for i in self._raw_val]
        self.ds_test = [self.ner_tags_to_spans(i) for i in self._raw_test]

        # Store original counts for logging
        orig_train = len(self.ds_train)
        orig_val = len(self.ds_val)
        orig_test = len(self.ds_test)

        if self.filter_empty_entities:
            # Filter out entries with no entities (only O tags)
            self.ds_train = self._filter_empty_entities(self.ds_train)
            self.ds_val = self._filter_empty_entities(self.ds_val)
            self.ds_test = self._filter_empty_entities(self.ds_test)

        print(f"Filtered entries with no entities:")
        print(f"  Train: {orig_train} -> {len(self.ds_train)} ({orig_train - len(self.ds_train)} removed)")
        print(f"  Val: {orig_val} -> {len(self.ds_val)} ({orig_val - len(self.ds_val)} removed)")
        print(f"  Test: {orig_test} -> {len(self.ds_test)} ({orig_test - len(self.ds_test)} removed)")
        self._print_entity_distribution(mode="spans")

    def _get_ner_tags(self):
        """
        Returns a sorted list of all unique entity types from the converted spans.
        
        Returns:
            list: Sorted list of unique entity types from all splits.
        """
        all_tags = set()
        
        for split in [self.ds_train, self.ds_val, self.ds_test]:
            for sample in split:
                for span in sample['ner']:
                    # Each span is (start, end, entity_type)
                    all_tags.add(span[2])
        
        return sorted(list(all_tags))

    def _get_ner_tags_raw(self):
        """
        Returns a sorted list of all unique entity types from BIO-tagged data.
        Strips B-/I- prefixes and excludes the 'O' tag.
        
        Returns:
            list: Sorted list of unique entity types from all splits.
        """
        all_tags = set()
        
        for split in [self.ds_train, self.ds_val, self.ds_test]:
            for sample in split:
                for tag in sample['ner_tags']:
                    if tag != 'O' and (tag.startswith('B-') or tag.startswith('I-')):
                        all_tags.add(self._normalize_label(tag[2:]))
        
        return sorted(list(all_tags))
    
    def _get_ner_tags_bio_raw(self):
        """
        Returns a sorted list of all unique entity types from BIO-tagged data.
        
        Returns:
            list: Sorted list of unique entity types from all splits.
        """
        all_tags = set()
        
        for split in [self.ds_train, self.ds_val, self.ds_test]:
            for sample in split:
                for tag in sample['ner_tags']:
                    if tag != 'O' and (tag.startswith('B-') or tag.startswith('I-')):
                        all_tags.add(self._normalize_label(tag[2:]))

        result = sorted(list(all_tags))

        expanded_candidates = []

        for label in result:
            expanded_candidates.append(f"b-{label}")
            expanded_candidates.append(f"i-{label}")

        return expanded_candidates

    def _print_entity_distribution(self, mode="spans"):
        """Print a table showing entity type counts across train/val/test splits.

        Args:
            mode: "spans" for converted span format, "raw" for BIO-tag format.
        """
        splits = {"Train": self.ds_train, "Val": self.ds_val, "Test": self.ds_test}
        counters = {}

        for split_name, data in splits.items():
            counter = Counter()
            for sample in data:
                if mode == "spans":
                    for span in sample["ner"]:
                        counter[span[2]] += 1
                else:
                    for tag in sample["ner_tags"]:
                        if tag.startswith("B-"):
                            counter[self._normalize_label(tag[2:])] += 1
            counters[split_name] = counter

        # Collect all entity types sorted by train count (most frequent first)
        all_types = sorted(
            {t for c in counters.values() for t in c},
            key=lambda t: counters["Train"][t],
            reverse=True
        )
        if not all_types:
            print("\nEntity distribution: no entities found.")
            return

        # Compute column widths
        type_width = max(len(t) for t in all_types)
        type_width = max(type_width, len("Entity Type"))
        num_width = 8

        col_width = num_width + 8  # number + space + (xx.x%)
        header = f"  {'Entity Type':<{type_width}}  {'Train':>{col_width}}  {'Val':>{col_width}}  {'Test':>{col_width}}  {'Total':>{col_width}}"
        separator = "  " + "-" * (type_width + 4 * (col_width + 2))

        print(f"\nEntity distribution across splits:")
        print(header)
        print(separator)

        # Compute totals per split for percentage calculation
        total_train = sum(counters["Train"].values())
        total_val = sum(counters["Val"].values())
        total_test = sum(counters["Test"].values())
        grand_total = total_train + total_val + total_test

        for entity_type in all_types:
            t = counters["Train"][entity_type]
            v = counters["Val"][entity_type]
            te = counters["Test"][entity_type]
            row_total = t + v + te
            t_pct = f"({t / total_train * 100:.1f}%)" if total_train else ""
            v_pct = f"({v / total_val * 100:.1f}%)" if total_val else ""
            te_pct = f"({te / total_test * 100:.1f}%)" if total_test else ""
            tot_pct = f"({row_total / grand_total * 100:.1f}%)" if grand_total else ""
            print(f"  {entity_type:<{type_width}}  {t:>{num_width}} {t_pct:>7}  {v:>{num_width}} {v_pct:>7}  {te:>{num_width}} {te_pct:>7}  {row_total:>{num_width}} {tot_pct:>7}")

        print(separator)
        print(f"  {'TOTAL':<{type_width}}  {total_train:>{num_width}} {'':>7}  {total_val:>{num_width}} {'':>7}  {total_test:>{num_width}} {'':>7}  {grand_total:>{num_width}} {'':>7}")

    def ner_tags_to_spans(self, samples):
        """
        Converts NER tags in the dataset samples to spans (start, end, entity type).
        
        Args:
            samples (dict): A dictionary containing the tokens and NER tags.
        
        Returns:
            dict: A dictionary containing tokenized text and corresponding NER spans.
        """
        ner_tags = samples["ner_tags"]
        spans = []
        start_pos = None
        entity_name = None

        for i, tag in enumerate(ner_tags):
            if tag == 'O':  # 'O' tag
                if entity_name is not None:
                    spans.append((start_pos, i - 1, entity_name))
                    entity_name = None
                    start_pos = None
            else:
                if tag.startswith('B-'):
                    if entity_name is not None:
                        spans.append((start_pos, i - 1, entity_name))
                    entity_name = self._normalize_label(tag[2:])  # Remove 'B-' prefix and normalize
                    start_pos = i
                elif tag.startswith('I-'):
                    continue  # Continue the current entity

        # Handle the last entity if the sentence ends with an entity
        if entity_name is not None:
            spans.append((start_pos, len(samples[self.text_column_name]) - 1, entity_name))
        
        return {"tokenized_text": samples[self.text_column_name], "ner": spans, "tags_present": self._extract_tags_present(ner_tags)}

    def get_tag_description(self, ds_description="disi-unibo-nlp/Pile-NER-biomed-descriptions"):
        """
        Load tag descriptions from a dataset and match them with self.ner_tags.
        
        Args:
            ds_description: Dataset name containing entity_type and description fields
            
        Returns:
            dict: Dictionary mapping tag names to their descriptions (None if not found)
        """
        # Load the dataset with tag descriptions
        # it must have format: {"entity_type": str, "description": str }
        if self.data_dir is not None:
            local_path = os.path.join(self.data_dir, self._name_to_slug(ds_description))
            print(f"  Loading descriptions from disk: {local_path}")
            ds = load_from_disk(local_path)
        else:
            ds = load_dataset(ds_description)
        
        # Get the actual data from the first available split
        if 'train' in ds:
            description_data = ds['train']
        elif len(ds.keys()) > 0:
            description_data = ds[list(ds.keys())[0]]
        else:
            raise ValueError(f"No data found in description dataset: {ds_description}")
        
        # Helper function to normalize tags
        def _normalize_tag(tag):
            # Remove special characters (keep only alphanumeric and spaces)
            normalized = re.sub(r'[^a-zA-Z0-9\s]', ' ', tag)
            # Convert multiple spaces to single space
            normalized = re.sub(r'\s+', ' ', normalized)
            # Convert to lowercase and strip whitespace
            return normalized.lower().strip()
        
        # Build lookup dictionary once (more efficient than searching for each tag)
        normalized_descriptions = {}
        for item in description_data:
            normalized_type = _normalize_tag(item['entity_type'])
            normalized_descriptions[normalized_type] = item['description']
        
        # Build result dictionary maintaining order of self.ner_tags
        self.tag_with_descriptions = {}
        for tag in self.ner_tags:
            normalized_tag = _normalize_tag(tag)
            # Fallback to the entity name itself when the catalog has no
            # description for this tag — keeps the description channel
            # label-specific instead of feeding a constant placeholder.
            self.tag_with_descriptions[tag] = normalized_descriptions.get(normalized_tag, tag)
        
        return self.tag_with_descriptions

    def remove_below_threshold(self, removal_threshold: int):
        """Remove entity types that appear fewer than removal_threshold times in the train split.

        Filters out those entity types from all splits (train/val/test) and updates self.ner_tags.
        Samples that end up with no entities after filtering are removed entirely.

        Args:
            removal_threshold: Minimum number of occurrences in train for an entity type to be kept.
        """
        # Determine mode by inspecting the first sample
        is_span_mode = 'ner' in self.ds_train[0] if self.ds_train else True

        # Count entity occurrences in train
        train_counter = Counter()
        if is_span_mode:
            for sample in self.ds_train:
                for span in sample['ner']:
                    train_counter[span[2]] += 1
        else:
            for sample in self.ds_train:
                for tag in sample['ner_tags']:
                    if tag.startswith('B-'):
                        train_counter[self._normalize_label(tag[2:])] += 1

        # Identify types to remove
        types_to_remove = {t for t, count in train_counter.items() if count < removal_threshold}
        # Also remove types that appear in val/test but not in train at all
        all_types = set(self.ner_tags)
        types_not_in_train = all_types - set(train_counter.keys())
        types_to_remove |= types_not_in_train

        if not types_to_remove:
            print(f"No entity types below threshold ({removal_threshold}). Nothing removed.")
            return

        types_to_keep = all_types - types_to_remove
        print(f"\nRemoving {len(types_to_remove)} entity type(s) below threshold ({removal_threshold} in train):")
        for t in sorted(types_to_remove):
            print(f"  - {t} (train count: {train_counter.get(t, 0)})")

        if is_span_mode:
            self.ds_train = self._filter_spans_by_types(self.ds_train, types_to_keep)
            self.ds_val = self._filter_spans_by_types(self.ds_val, types_to_keep)
            self.ds_test = self._filter_spans_by_types(self.ds_test, types_to_keep)
        else:
            self.ds_train = self._filter_raw_by_types(self.ds_train, types_to_remove)
            self.ds_val = self._filter_raw_by_types(self.ds_val, types_to_remove)
            self.ds_test = self._filter_raw_by_types(self.ds_test, types_to_remove)

        self.ner_tags = sorted(types_to_keep)
        if hasattr(self, 'ner_tags_bio'):
            self.ner_tags_bio = []
            for label in self.ner_tags:
                self.ner_tags_bio.append(f"b-{label}")
                self.ner_tags_bio.append(f"i-{label}")

        print(f"\nAfter removal:")
        print(f"  Entity types: {len(self.ner_tags)} remaining")
        print(f"  Train: {len(self.ds_train)} samples")
        print(f"  Val: {len(self.ds_val)} samples")
        print(f"  Test: {len(self.ds_test)} samples")

    def _filter_spans_by_types(self, data, types_to_keep):
        """Filter span-format data, keeping only spans with entity types in types_to_keep."""
        filtered = []
        for sample in data:
            new_spans = [s for s in sample['ner'] if s[2] in types_to_keep]
            if new_spans:
                new_tags_present = sorted({s[2] for s in new_spans})
                filtered.append({**sample, 'ner': new_spans, 'tags_present': new_tags_present})
        return filtered

    def _filter_raw_by_types(self, data, types_to_remove):
        """Filter raw BIO-tag data, converting removed entity types to 'O'."""
        filtered = []
        for sample in data:
            new_tags = []
            for tag in sample['ner_tags']:
                if tag != 'O' and (tag.startswith('B-') or tag.startswith('I-')):
                    entity_type = self._normalize_label(tag[2:])
                    if entity_type in types_to_remove:
                        new_tags.append('O')
                    else:
                        new_tags.append(tag)
                else:
                    new_tags.append(tag)
            if any(t != 'O' for t in new_tags):
                filtered.append({
                    **sample,
                    'ner_tags': new_tags,
                    'tags_present': self._extract_tags_present(new_tags),
                    'tag_present_bio': self._extract_tags_present_bio(new_tags),
                })
        return filtered