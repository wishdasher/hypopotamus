#!/usr/bin/env python3 -*- coding: utf-8 -*-

"""
Hypopotamus Wikipedia Parser: Create a triplets file from the Wikipedia dump.

The wikipedia-json.zip can be downloaded at:


The enwiki-latest-pages-articles.xml.bz2 can be downloaded at:
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

Usage:
  wiki2triplets.py --help
  wiki2triplets.py --fromjson <inputfile> [--maxpathlen <pathlen>] [--output <outputfile>]
  wiki2triplets.py --fromdump <inputfile> [--maxpathlen <pathlen>] [--output <outputfile>]


Options:
  -h --help                     Show this screen.
  -j, --fromjson <inputfile>    Uses the annotated Wikipedia JSON file as input.
  -d, --fromdump <inputfile>    Uses the raw Wikipedia dump as input.
  -o, --output <outputfile>     Option to output to file
  -m, --maxpathlen <pathlen>    Maximum path len between the entities considered for extraction [default: 4].


Try:
  python3 wiki2triplets.py --fromjson wikipedia-json.zip
  python3 wiki2triplets.py --fromjson wikipedia-json.zip --maxpathlen 5 --output triplets.txt
  python3 wiki2triplets.py --fromjson wikipedia-json.zip -m 5 -o triplets.txt
  python3 wiki2triplets.py --fromdump enwiki-latest-pages-articles.xml.bz2
  python3 wiki2triplets.py --fromdump enwiki-latest-pages-articles.xml.bz2 -m 5 -o triplets.txt
"""

import io
import os
import sys
import json
from zipfile import ZipFile

from docopt import docopt

from spacy.en import English
from spacy.tokens.token import Token as SpacyToken


def extract_paths_between_nouns(sentence):
    """
    Get all the dependency paths between nouns in the sentence

    :param sentence: the sentence to parse
    :return: a list of entities and paths, each element a list of X, Y, path
    """

    all_nouns = [(token, i, i) for i, token in enumerate(sentence)
                 if token.tag_.startswith('NN') and len(token.string.strip()) > 1]
    # Extending list of entities for noun chunks
    # TODO consider, edit, test, should articles and pronouns be counted in noun chunks
    # also, this returns all noun chunks from the entire document
    noun_chunks = [(np, np.start, np.end) for np in sentence.doc.noun_chunks]
    all_nouns.extend(noun_chunks)

    # Extract all dependency paths between nouns, up to length 4
    # TODO consider that this does not include adjacent noun pairs
    pairs = [(x[0], y[0]) for x in all_nouns for y in all_nouns if x[2] < y[1]]
    paths = [path for path in map(shortest_path, pairs) if path is not None]
    paths = [p for path in paths for p in get_satellite_links(path)]
    paths = [path for path in map(clean_path, paths) if path is not None]

    return paths


def shortest_path(tokens):
    """
    Returns the shortest dependency path from x to y

    :param tokens: a tuple (x_token, y_token)
    :return: the shortest dependency path from x to y
             in format (x, path from x to lch, lowest common head, path from lch to y, y)
    """

    x, y = tokens

    x_token = x
    y_token = y

    # If one of the tokens is a Span (noun chunk), find the root
    if not isinstance(x_token, SpacyToken):
        x_token = x_token.root
    if not isinstance(y_token, SpacyToken):
        y_token = y_token.root

    # Get the path from the root to each of the tokens
    hx = heads(x_token)
    hy = heads(y_token)

    # TODO is this even right?
    # Get the lowest common head
    i = -1
    for i in range(min(len(hx), len(hy))):
        if hx[i] is not hy[i]:
            break

    if i == -1:
        lch_idx = 0
        if len(hy) > 0:
            lch = hy[lch_idx]
        elif len(hx) > 0:
            lch = hx[lch_idx]
        else:
            lch = None
    elif hx[i] == hy[i]:
        lch_idx = i
        lch = hx[lch_idx]
    else:
        lch_idx = i-1
        lch = hx[lch_idx]

    # The path from x to the lowest common head
    hx = hx[lch_idx+1:]
    if lch and check_direction(lch, hx, lambda h: h.lefts):
        return None
    hx = hx[::-1]

    # The path from the lowest common head to y
    hy = hy[lch_idx+1:]
    if lch and check_direction(lch, hy, lambda h: h.rights):
        return None

    return (x, hx, lch, hy, y)


def heads(token):
    """
    Return the heads of a token, from the root down to immediate head

    :param token: spacy token
    :return: the heads of the token
    """
    hs = []
    while token is not token.head:
        token = token.head
        hs.append(token)
    return hs[::-1]


def check_direction(lch, hs, f_dir):
    """
    Make sure that the path between the term and the lowest common head is in a
    certain direction

    :param lch: the lowest common head
    :param hs: the path from the lowest common head to the term
    :param f_dir: function of direction
    :return:
    """
    # TODO why is this important
    return any(modifier not in f_dir(head) for head, modifier in zip([lch] + hs[:-1], hs))


def get_satellite_links(path):
    """
    Add the "setallites" - single links not already contained in the dependency
    path added on either side of each noun

    :param x: the X token
    :param y: the Y token
    :param hx: X's head tokens
    :param hy: Y's head tokens
    :param lch: the lowest common ancestor of X and Y
    :return: more paths, with satellite links
    """

    # TODO: only adds on one dependency edge (first one, why?), doesn't consider x.rights, y.lefts\
    x, hx, lch, hy, y = path
    paths = [(None, x, hx, lch, hy, y, None)]

    x_lefts = list(x.lefts)
    if len(x_lefts) > 0:
        paths.append((x_lefts[0], x, hx, lch, hy, y, None))

    y_rights = list(y.rights)
    if len(y_rights) > 0:
        paths.append((None, x, hx, lch, hy, y, y_rights[0]))

    return paths


def edge_to_string(token, is_head=False):
    """
    Converts the token to an edge string representation
    :param token: the token
    :return: the edge string
    """
    t = token
    if not isinstance(token, SpacyToken):
        t = token.root

    return '/'.join([token_to_lemma(token), t.pos_, t.dep_
                     if t.dep_ != '' and not is_head else 'ROOT'])


def argument_to_string(token, edge_name):
    """
    Converts the argument token (X or Y) to an edge string representation
    :param token: the X or Y token
    :param edge_name: 'X' or 'Y'
    :return:
    """
    if not isinstance(token, SpacyToken):
        token = token.root

    return '/'.join([edge_name, token.pos_, token.dep_
                     if token.dep_ != '' else 'ROOT'])


def direction(dir):
    """
    Print the direction of the edge
    :param dir: the direction
    :return: a string representation of the direction
    """
    # Up to the head
    if dir == UP:
        return '>'
    # Down from the head
    elif dir == DOWN:
        return '<'


def token_to_string(token):
    """
    Convert the token to string representation
    :param token:
    :return:
    """
    if not isinstance(token, SpacyToken):
        return ' '.join([t.string.strip().lower() for t in token])
    else:
        return token.string.strip().lower()


def token_to_lemma(token):
    """
    Convert the token to string representation
    :param token: the token
    :return: string representation of the token
    """
    if not isinstance(token, SpacyToken):
        return token_to_string(token)
    else:
        return token.lemma_.strip().lower()


def clean_path(path):
    """
    Filter out long paths and pretty print the short ones
    :return: the string representation of the path
    """

    set_x, x, hx, lch, hy, y, set_y = path
    set_path_x = []
    set_path_y = []
    lch_lst = []

    if set_x:
        set_path_x = [edge_to_string(set_x) + direction(DOWN)]
    if set_y:
        set_path_y = [direction(UP) + edge_to_string(set_y)]

    # X is the head
    if isinstance(x, SpacyToken) and lch == x:
        dir_x = ''
        dir_y = direction(DOWN)
    # Y is the head
    elif isinstance(y, SpacyToken) and lch == y:
        dir_x = direction(UP)
        dir_y = ''
    # X and Y are not heads
    else:
        lch_lst = [edge_to_string(lch, is_head=True)] if lch else []
        dir_x = direction(UP)
        dir_y = direction(DOWN)

    len_path = len(hx) + len(hy) + len(set_path_x) + len(set_path_y) + len(lch_lst)

    if len_path <= MAX_PATH_LEN:
        cleaned_path = '_'.join(set_path_x + [argument_to_string(x, 'X') + dir_x] +
                                [edge_to_string(token) + direction(UP) for token in hx] +
                                lch_lst +
                                [direction(DOWN) + edge_to_string(token) for token in hy] +
                                [dir_y + argument_to_string(y, 'Y')] + set_path_y)
        return token_to_string(x), token_to_string(y), cleaned_path
    else:
        return None


def iter_paragraph(arguments):
    """
    A helper function to iterate through the two types of Wikipedia data inputs.

    :param arguments: The docopt arguments
    :type arguments: dict
    :return: A generator yielding a pargraph of text for each iteration.
    """
    # Iterating through paragraphes from the Anntoated Wikipedia zipfile.
    if arguments['--fromjson']:
        with ZipFile(arguments['--fromjson'], 'r') as zip_in:
            # Iterate through the individual files.
            for infile in zip_in.namelist():
                if infile.endswith('/'): # Skip the directories.
                    continue
                with zip_in.open(infile) as f_in:
                    for line in io.TextIOWrapper(f_in, 'utf8'):
                        # Each line is a separate json.
                        data = json.loads(line)
                        # The useful text under 'text' key.
                        paragraph = data['text'].strip()
                        yield paragraph
    # Iterating through paragraphes from the Wikipedia dump.
    elif '--fromdump' in arguments.keys() and arguments['--fromdump']:
        infile = arguments['--fromdump']
        # Simply iterate through every line in the dump
        # and treat each line as a paragraph.
        with io.open(infile, 'r', encoding='utf8') as f_in:
            for paragraph in f_in:
                if pargraph:
                    yield paragraph


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Hypopotamus (wiki2triplets.py) version 0.0.1')
    print (arguments)
    if arguments['--output']: # If output is specified, print output to a file.
        outfile = arguments['--output']
        f_out = io.open(outfile, 'w', encoding='utf8')
    else: # Else, print output to the console.
        f_out = sys.stdout

    # Check that the user used an integer for the maxpathlen option.
    assert arguments['--maxpathlen'].isdigit(), "--maxpathlen option only takes integer!!"
    # Initialize the MAX_PATH_LEN variable.
    MAX_PATH_LEN = int(arguments['--maxpathlen'])

    # Set the integer for UP and DOWN ???
    # Actually are they just used as some strange symbol?
    UP = 1
    DOWN = 2

    # Initialize the Spacy Parser.
    annotator = English()

    # Iterate through the pargraphs in Wikipedia articles.
    for paragraph in iter_paragraph(arguments):
        annotated_paragraph = annotator(paragraph)
        # Extract the path for each sentence separately
        for sentence in annotated_paragraph.sents:
            paths = extract_paths_between_nouns(sentence)
            if paths:
                for path in paths:
                    print('\t'.join(path), end='\n', file=f_out)
