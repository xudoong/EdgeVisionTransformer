import tqdm
import csv
import os

from logger import logger
import torch
from torch.utils.data import TensorDataset
from dependency_parser import parse_sent
from util import none_if_empty
import classifier_scoring as scoring


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For
                single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        self.parse_a = None
        self.parse_b = None


def add_dependency_arcs(examples, verbose=True):
    if isinstance(examples, InputExample):
        examples = [examples]
    # Do text_a
    for example in tqdm.tqdm(examples, desc="Parsing", disable=not verbose):
        example.parse_a = parse_sent(example.text_a)
        if example.text_b is not None:
            example.parse_b = parse_sent(example.text_b)
    return examples


class NLIDiagnosticExample(InputExample):

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        lex_sem=None,
        pred_arg_struct=None,
        logic=None,
        knowledge=None,
        domain=None,
    ):
        super(NLIDiagnosticExample, self).__init__(guid, text_a, text_b, label)
        self.lex_sem = lex_sem
        self.pred_arg_struct = pred_arg_struct
        self.logic = logic
        self.knowledge = knowledge
        self.domain = domain


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def get_dummy_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self.get_train_examples(data_dir)[:20]

    def get_dummy_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dev_examples(data_dir)[:10]

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @property
    def scorer(self):
        """Return a scoring class (accuracy, F1, etc...)"""
        return scoring.Accuracy()


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples

    @property
    def scorer(self):
        """Return a scoring class (accuracy, F1, etc...)"""
        return scoring.F1()


class Sst2Processor(DataProcessor):
    """Processor for the binary SST data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{i}"
            text_a = line[0]
            label = line[1]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=None,
                label=label
            )
            examples.append(example)
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label
            )
            examples.append(example)
        return examples

class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = line[3]
            label = line[1]
            example = InputExample(
                guid=guid,
                text_a=text_a,
                text_b=None,
                label=label
            )
            examples.append(example)
        return examples

    @property
    def scorer(self):
        """Return a scoring class (accuracy, F1, etc...)"""
        return scoring.Matthews()


class DiagnosticProcessor(DataProcessor):
    """Processor for the GLUE diagnostic dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        raise ValueError("You can't train on the diagnostic data naughty boy")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "diagnostic-full.tsv")),
            "diagnostic")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            example = NLIDiagnosticExample(
                guid=f"{set_type}-{i}",
                text_a=line[5],
                text_b=line[6],
                label=line[7],
                lex_sem=none_if_empty(line[0]),
                pred_arg_struct=none_if_empty(line[1]),
                logic=none_if_empty(line[2]),
                knowledge=none_if_empty(line[3]),
                domain=none_if_empty(line[4]),
            )
            examples.append(example)
        return examples

    def get_dummy_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.get_dev_examples(data_dir)[:10]


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    verbose=True,
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        # tokens: [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        # ids:    0     0  0    0    0     0       0 0     1  1  1  1   1 1
        #
        # (b) For single sequences:
        # tokens:   [CLS] the dog is hairy . [SEP]
        # ids: 0   0   0   0  0     0 0
        #
        # Where "ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0`
        # and `type=1` were learned during pre-training and are added to the
        # wordpiece embedding vector (and position vector). This is not
        # *strictly* necessary since the [SEP] token unambigiously separates
        # the sequences, but it makes it easier for the model to learn the
        # concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS])
        # is used as as the "sentence vector". Note that this only makes sense
        # because the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then
    # each token that's truncated likely contains more information than a
    # longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def prepare_tensor_dataset(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    verbose=True,
):
    features = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, verbose)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    tensor_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return tensor_data


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mis": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
}

num_labels_task = {
    "cola": 2,
    "mnli": 3,
    "mnli-mis": 3,
    "mrpc": 2,
    "sst-2": 2,
}


def is_nli_task(processor):
    is_mnli = isinstance(processor, MnliProcessor)
    is_diagnostic = isinstance(processor, DiagnosticProcessor)
    return is_mnli or is_diagnostic
