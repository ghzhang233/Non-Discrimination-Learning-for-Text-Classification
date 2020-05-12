"""A small, simple tool for generating eval data to detect bias.
Using templates and lists of identity terms, this generates many comments that
we think should get similar scores from our toxicity model. If it turns out that
"ziv is a transgender factory worker" gets a significantly different score than
"joe is a tall computer programmer", then our model is exhibiting a
disproportionate adverse effect on certain groups.
This tool has no pretensions of being a real generative language model. It's
purpose is simply to generate sanity-test/evaluation data.
Example usage:
  $ python bias_madlibs.py -num_examples 100
Outputs a CSV with headers "Label", "Text", and "Template".
Label values are "offensive" and "non_offensive". "Template" is an ID for the template to
allow grouping results by template.
"""

import argparse
import os
import random

from data_utils import get_sensitive_words


def _read_word_list(bias_data_dir, filename):
    """Reads a terms list and returns a list of strings, one per line."""
    with open(os.path.join(bias_data_dir, filename), "r", encoding="utf-8") as f:
        return f.read().splitlines()


class Madlibber(object):
    """Emits madlibs according to a set of templates."""

    # A list of template pairs that use the same word types and roughly similar
    # structure, but one is toxic and the other is neutral.
    TEMPLATE_PAIRS = [
        {
            'non_offensive': u'{verb_positive} {adjective_people}',
            'offensive': u'{verb_negative} {adjective_people}',
            'template': 'verb_adj'
        },
        {
            'non_offensive': u'i am {adjective_people}',
            'offensive': u'i hate {adjective_people}',
            'template': 'am_hate_adj'
        },
        {
            'non_offensive': u'Being {adjective_people} is {adjective_positive}.',
            'offensive': u'Being {adjective_people} is {adjective_negative}.',
            'template': 'being_adj'
        },
        {
            'non_offensive': u'You are a {adjective_positive} {adjective_people}',
            'offensive': u'You are a {adjective_negative} {adjective_people}',
            'template': 'you_are_adj'
        },
    ]

    def __init__(self, bias_data_dir):
        sensitive_words, sensitive_general_words, sensitive_extra_words = get_sensitive_words()
        self.sensitive_male_words = [sensitive_words[i] for i in range(len(sensitive_words)) if i % 2 == 0]
        self.sensitive_female_words = [sensitive_words[i] for i in range(len(sensitive_words)) if i % 2 == 1]
        self._template_choices = [
            # ('adjective_people', _read_word_list(bias_data_dir, 'adjectives_people.txt')),
            ('adjective_people', sensitive_general_words),
            # ('adjective_people', ["man", "woman", "male", "female", "boy", "girl", "son", "daughter", "brother",
            # "sister", "uncle", "aunt", "actor", "actress", "waiter", "waitress"]),
            ('adjective_positive', _read_word_list(bias_data_dir, 'adjectives_positive.txt')),
            ('adjective_negative', _read_word_list(bias_data_dir, 'adjectives_negative.txt')),
            ('verb_positive', _read_word_list(bias_data_dir, 'verbs_positive.txt')),
            ('verb_negative', _read_word_list(bias_data_dir, 'verbs_negative.txt')),
        ]
        self._filler_text = _read_word_list(bias_data_dir, 'filler.txt')

    def expand_template(self, template, add_filler):
        """Expands the template with randomly chosen words."""
        parts = {}
        for template_key, choices in self._template_choices:
            parts[template_key] = random.choice(choices)
        gender = "male" if parts["adjective_people"] in self.sensitive_male_words else "female"
        expanded = template.format(**parts)
        if add_filler:
            return u'{} {}'.format(expanded, random.choice(self._filler_text)), gender
        return expanded, gender


def _parse_args():
    """Returns parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-num_examples',
        type=int,
        default=50,
        help='Number of phrases to output (estimate).')
    parser.add_argument(
        '-bias_data_dir',
        type=str,
        default='data',
        help='Directory for bias data.')
    parser.add_argument(
        '-label',
        default='both',
        choices=['both', 'offensive', 'non_offensive'],
        help='Type of examples to output.')
    parser.add_argument(
        '-longer', action='store_true', help='Output longer phrases.')
    return parser.parse_args()


def _main():
    """Prints some madlibs."""
    args = _parse_args()
    madlibber = Madlibber(args.bias_data_dir)
    examples_per_template = max(
        1, args.num_examples // len(madlibber.TEMPLATE_PAIRS))
    example_set = set()

    def actual_label():
        if args.label in ('offensive', 'non_offensive'):
            return args.label
        else:
            return random.choice(('offensive', 'non_offensive'))

    with open("processed_data/madlib.csv", "w", encoding="utf-8") as fout:
        fout.write('text,label,gender,template\n')
        print('text,label,gender,template')
        for template_pair in madlibber.TEMPLATE_PAIRS:
            template_count = 0
            template_attempts = 0
            while template_count < examples_per_template and template_attempts < 7 * examples_per_template:
                template_attempts += 1
                label = actual_label()
                example, gender = madlibber.expand_template(template_pair[label], args.longer)
                if example not in example_set:
                    example_set.add(example)
                    template_count += 1
                    fout.write(u'{},{},{},{}\n'.format(example, label, gender, template_pair['template']))
                    print(u'{},{},{},{}'.format(example, label, gender, template_pair['template']))


if __name__ == '__main__':
    _main()
