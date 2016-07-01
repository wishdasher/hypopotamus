import io
import os
import sys
import spacy
import json

from spacy.en import English

def main():
    """Create a triplets file from the Wikipedia dump
    :output: a file with all the dependency paths, formatted as X\tY\tpath
    """

    parser = English()
    path_name = sys.argv[-1]

    for root, dirs, files in os.walk(path_name):
        for wiki_file in files:
            with io.open(os.path.join(root, wiki_file), 'r', encoding='utf8') as f_in:
                # Each line is a separate json, useful text under 'text'
                for line in f_in:
                    data = json.loads(line)

                    paragraph = (data['text']).strip()
                    if len(paragraph) == 0:
                        continue
                    parsed_paragraph = parser(paragraph)

                    # Parse each sentence separately
                    for sentence in parsed_paragraph.sents:
                        paths = parse_sentence(sentence)
                        if len(paths) > 0:
                            output = '\n'.join(['\t'.join(path) for path in paths])
                            print(output)

def parse_sentence(sentence):
    """Get all the dependency paths between nouns in the sentence
    :param sentence: the sentence to parse
    :return: a list of entities and paths, each element a list of X, Y, path
    """

    all_nouns = [(token, i, i) for i, token in enumerate(sentence)
                    if token.tag_[:2] == 'NN' and len(token.string.strip()) > 1]
    # Extending list of entities for noun chunks TODO consider, edit, test, should articles and pronouns be counted in noun chunks
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
    """Returns the shortest dependency path from x to y
    :param tokens: a tuple (x_token, y_token)
    :return: the shortest dependency path from x to y
             in format (x, path from x to lch, lowest common head, path from lch to y, y)
    """

    x, y = tokens

    x_token = x
    y_token = y

    # If one of the tokens is a Span (noun chunk), find the root
    if not isinstance(x_token, spacy.tokens.token.Token):
        x_token = x_token.root
    if not isinstance(y_token, spacy.tokens.token.Token):
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
    """Return the heads of a token, from the root down to immediate head
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
    Make sure that the path between the term and the lowest common head is in a certain direction
    :param lch: the lowest common head
    :param hs: the path from the lowest common head to the term
    :param f_dir: function of direction
    :return:
    """
    # TODO why is this important
    return any(modifier not in f_dir(head) for head, modifier in zip([lch] + hs[:-1], hs))


def get_satellite_links(path):
    """
    Add the "setallites" - single links not already contained in the dependency path added on either side of each noun
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
    if not isinstance(token, spacy.tokens.token.Token):
        t = token.root

    return '/'.join([token_to_lemma(token), t.pos_, t.dep_ if t.dep_ != '' and not is_head else 'ROOT'])


def argument_to_string(token, edge_name):
    """
    Converts the argument token (X or Y) to an edge string representation
    :param token: the X or Y token
    :param edge_name: 'X' or 'Y'
    :return:
    """
    if not isinstance(token, spacy.tokens.token.Token):
        token = token.root

    return '/'.join([edge_name, token.pos_, token.dep_ if token.dep_ != '' else 'ROOT'])


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
    if not isinstance(token, spacy.tokens.token.Token):
        return ' '.join([t.string.strip().lower() for t in token])
    else:
        return token.string.strip().lower()


def token_to_lemma(token):
    """
    Convert the token to string representation
    :param token: the token
    :return: string representation of the token
    """
    if not isinstance(token, spacy.tokens.token.Token):
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
    if isinstance(x, spacy.tokens.token.Token) and lch == x:
        dir_x = ''
        dir_y = direction(DOWN)
    # Y is the head
    elif isinstance(y, spacy.tokens.token.Token) and lch == y:
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


# Constants
MAX_PATH_LEN = 4
UP = 1
DOWN = 2

if __name__ == '__main__':
    # This is the main function
    main()