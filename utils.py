"""
Standard functions used to support relational learning
"""
from random import uniform

from fo_planner import Operator
from fo_planner import build_index
# from planners.fo_planner import subst
from fo_planner import is_variable
from fo_planner import extract_strings


def weighted_choice(choices):
    """
    A weighted version of choice.
    """
    total = sum(w for w, c in choices)
    r = uniform(0, total)
    upto = 0
    for w, c in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def get_variablizations(literal):
    for i, ele in enumerate(literal[1:]):
        if isinstance(ele, tuple):
            for inner in get_variablizations(ele):
                yield tuple([literal[0]] + [inner if j == i else iele for j,
                                            iele in enumerate(literal[1:])])
        elif not is_variable(ele):
            yield tuple([literal[0]] + ['?gen%i' % hash(ele) if j == i else
                                        iele for j, iele in
                                        enumerate(literal[1:])])


def count_occurances(var, h):
    return len([s for x in h for s in extract_strings(x) if s == var])


def test_coverage(h, constraints, pset, nset):
    new_pset = [(p, pm) for p, pm in pset if
                covers(h.union(constraints), p, pm)]
    new_nset = [(n, nm) for n, nm in nset if
                covers(h.union(constraints), n, nm)]
    return new_pset, new_nset


def covers(h, x, initial_mapping):
    """
    Returns true if h covers x
    """
    index = build_index(x)
    operator = Operator(tuple(['Rule']), h, [])
    for m in operator.match(index, initial_mapping=initial_mapping):
        return True
    return False


def rename(mapping, literal):
    """
    Given a mapping, renames the literal. Unlike subst, this works with
    constants as well as variables.
    """
    return tuple(mapping[ele] if ele in mapping else rename(mapping, ele) if
                 isinstance(ele, tuple) else ele for ele in literal)


def generate_literal(relation, arity, gensym):
    """
    Returns a new literal with novel variables.
    """
    return (relation,) + tuple(gensym() for i in range(arity))


def generalize_literal(literal, gensym):
    """
    This takes a literal and returns the most general version of it possible.
    i.e., a version that has all the values replaced with new veriables.
    """
    return (literal[0],) + tuple(ele if is_variable(ele) else
                                 # '?gen%s' % hash(ele)
                                 gensym()
                                 for ele in literal[1:])


def remove_vars(literal):
    """
    This removes all variables by putting XXX at the front of the string, so it
    cannot be unified anymore.
    """
    return tuple('XXX' + ele if is_variable(ele) else remove_vars(ele) if
                 isinstance(ele, tuple) else ele for ele in literal)


def clause_length(clause):
    """
    Counts the length of a clause. In particular, it counts number of
    relations, constants, and variable equality relations.
    """
    var_counts = {}
    count = 0

    for l in clause:
        count += count_elements(l, var_counts)

    for v in var_counts:
        count += var_counts[v] - 1

    return count


def count_elements(x, var_counts):
    """
    Counts the number of constants and keeps track of variable occurnaces.
    """
    c = 0
    if isinstance(x, tuple):
        c = sum([count_elements(ele, var_counts) for ele in x])
    elif is_variable(x):
        if x not in var_counts:
            var_counts[x] = 0
        var_counts[x] += 1
    else:
        c = 1
    return c
